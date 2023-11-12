#!/usr/bin/env python3
# This is free and unencumbered software released into the public domain. For more detail,
# see the LICENCE file at https://github.com/adefossez/seewav
# Original author: adefossez
"""
Generates a nice waveform visualization from an audio file, save it as a mp4 file.
"""
import os
from colorthief import ColorThief
from PIL import Image
import colorsys
import base64
import tqdm
import numpy as np
import cairo
from gi.repository import Rsvg
import argparse
import json
import math
import subprocess as sp
import sys
import tempfile
from pathlib import Path

import gi
gi.require_version('Rsvg', '2.0')


_is_main = False


def colorize(text, color):
    """
    Wrap `text` with ANSI `color` code. See
    https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def fatal(msg):
    """
    Something bad happened. Does nothing if this module is not __main__.
    Display an error message and abort.
    """
    if _is_main:
        head = "error: "
        if sys.stderr.isatty():
            head = colorize("error: ", 1)
        print(head + str(msg), file=sys.stderr)
        sys.exit(1)


def read_info(media):
    """
    Return some info on the media file.
    """
    proc = sp.run([
        'ffprobe', "-loglevel", "panic",
        str(media), '-print_format', 'json', '-show_format', '-show_streams'
    ],
        capture_output=True)
    if proc.returncode:
        raise IOError(f"{media} does not exist or is of a wrong type.")
    return json.loads(proc.stdout.decode('utf-8'))


def read_audio(audio, seek=None, duration=None):
    """
    Read the `audio` file, starting at `seek` (or 0) seconds for `duration` (or all)  seconds.
    Returns `float[channels, samples]`.
    """

    info = read_info(audio)
    channels = None
    stream = info['streams'][0]
    if stream["codec_type"] != "audio":
        raise ValueError(f"{audio} should contain only audio.")
    channels = stream['channels']
    samplerate = float(stream['sample_rate'])

    # Good old ffmpeg
    command = ['ffmpeg', '-y']
    command += ['-loglevel', 'panic']
    if seek is not None:
        command += ['-ss', str(seek)]
    command += ['-i', audio]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-f', 'f32le']
    command += ['-']

    proc = sp.run(command, check=True, capture_output=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return wav.reshape(-1, channels).T, samplerate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def envelope(wav, window, stride):
    """
    Extract the envelope of the waveform `wav` (float[samples]), using average pooling
    with `window` samples and the given `stride`.
    """
    # pos = np.pad(np.maximum(wav, 0), window // 2)
    wav = np.pad(wav, window // 2)
    out = []
    for off in range(0, len(wav) - window, stride):
        frame = wav[off:off + window]
        out.append(np.maximum(frame, 0).mean())
    out = np.array(out)
    # Some form of audio compressor based on the sigmoid.
    out = 1.9 * (sigmoid(2.5 * out) - 0.5)
    return out


def draw_env(envs, out, fg_colors, bg_color, size, image):
    """
    Internal function, draw a single frame (two frames for stereo) using cairo and save
    it to the `out` file as png. envs is a list of envelopes over channels, each env
    is a float[bars] representing the height of the envelope to draw. Each entry will
    be represented by a bar.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1920, 1080)
    ctx = cairo.Context(surface)

    svg = Rsvg.Handle.new_from_data(image.encode("utf-8"))
    svg.render_cairo(ctx)
    ctx.scale(*size)

    K = len(envs)  # Number of waves to draw (waves are stacked vertically)
    T = len(envs[0])  # Numbert of time steps
    pad_ratio = 0.1  # spacing ratio between 2 bars
    base_width = 0.86302083333
    left_pad = (1 - base_width) / 2
    width = base_width / (T * (1 + 2 * pad_ratio))
    pad = pad_ratio * width
    delta = 2 * pad + width

    ctx.set_line_width(width)
    for step in range(T):
        for i in range(K):
            half = 0.5 * envs[i][step]  # (semi-)height of the bar
            half /= K  # as we stack K waves vertically
            midrule = 0.95  # midrule of i-th wave
            ctx.set_source_rgb(*fg_colors[i])
            ctx.move_to(pad + step * delta + left_pad, midrule - half)
            ctx.line_to(pad + step * delta + left_pad, midrule)
            ctx.stroke()
            # ctx.set_source_rgba(*fg_colors[i], 0.8)
            # ctx.move_to(pad + step * delta, midrule)
            # ctx.line_to(pad + step * delta, midrule + 0.9 * half)
            # ctx.stroke()

    surface.write_to_png(out)


def interpole(x1, y1, x2, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def visualize(audio,
              tmp,
              out,
              seek=None,
              duration=None,
              rate=60,
              bars=50,
              speed=4,
              time=0.4,
              oversample=3,
              fg_color=(.2, .2, .2),
              fg_color2=(.5, .3, .6),
              bg_color=(1, 1, 1),
              size=(400, 400),
              stereo=False,
              cover_path="cover.png",
              song_data={
        "title": "title",
        "artist": "artist",
        "date": "date",
        "label": "label"
                  }
              ):
    """
    Generate the visualisation for the `audio` file, using a `tmp` folder and saving the final
    video in `out`.
    `seek` and `durations` gives the extract location if any.
    `rate` is the framerate of the output video.

    `bars` is the number of bars in the animation.
    `speed` is the base speed of transition. Depending on volume, actual speed will vary
        between 0.5 and 2 times it.
    `time` amount of audio shown at once on a frame.
    `oversample` higher values will lead to more frequent changes.
    `fg_color` is the rgb color to use for the foreground.
    `fg_color2` is the rgb color to use for the second wav if stereo is set.
    `bg_color` is the rgb color to use for the background.
    `size` is the `(width, height)` in pixels to generate.
    `stereo` is whether to create 2 waves.
    """
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        fatal(err)
        raise
    # wavs is a list of wav over channels
    wavs = []
    if stereo:
        assert wav.shape[0] == 2, 'stereo requires stereo audio file'
        wavs.append(wav[0])
        wavs.append(wav[1])
    else:
        wav = wav.mean(0)
        wavs.append(wav)

    for i, wav in enumerate(wavs):
        wavs[i] = wav/wav.std()

    window = int(sr * time / bars)
    stride = int(window / oversample)
    # envs is a list of env over channels
    envs = []
    for wav in wavs:
        env = envelope(wav, window, stride)
        env = np.pad(env, (bars // 2, 2 * bars))
        envs.append(env)

    duration = len(wavs[0]) / sr
    frames = int(rate * duration)
    smooth = np.hanning(bars)

    print("Generating the frames...")

    svg = ""
    with open("base.svg", "r") as f:
        svg = f.read()

    color_thief = ColorThief(cover_path)
    dominant_color = color_thief.get_palette(color_count=10)
    dom_col_hsl = []
    for x in dominant_color:
        dom_col_hsl.append(colorsys.rgb_to_hsv(x[0], x[1], x[2]))
    print(dominant_color)
    print(dom_col_hsl)
    hue = 0
    averageSaturation = 0
    found = False
    for x in dom_col_hsl:
        averageSaturation += x[1]
        if x[1] > 0.1 and found == False:
            found = True
            print("hue is " + str(x[0]*255))
            hue = x[0]*255

    averageSaturation = averageSaturation / 10
    print("average saturation: " + str(averageSaturation))
    if averageSaturation > 0.1:
        svg = svg.replace("var(--backdrop)", "hsl("+str(hue)+", 36%, 22%)")
        svg = svg.replace("var(--lower)", "hsl("+str(hue)+", 36%, 15%)")
        svg = svg.replace("var(--bar)", "hsl("+str(hue)+", 36%, 31%)")
        svg = svg.replace("var(--text)", "hsl("+str(hue)+", 100%, 96%)")
    else:
        svg = svg.replace("var(--backdrop)", "hsl("+str(hue)+", 0%, 22%)")
        svg = svg.replace("var(--lower)", "hsl("+str(hue)+", 0%, 15%)")
        svg = svg.replace("var(--bar)", "hsl("+str(hue)+", 0%, 31%)")
        svg = svg.replace("var(--text)", "hsl("+str(hue)+", 0%, 96%)")
    cover = ""

    im = Image.open(cover_path)
    im.thumbnail((404, 404), Image.Resampling.LANCZOS)
    im.save("cover-compressed.png", "PNG")
    with open('cover-compressed.png', 'rb') as imagefile:
        cover = base64.b64encode(imagefile.read()).decode('ascii')

    imagedata = "data:image/png;base64,"
    imagedata += cover

    svg = svg.replace("IMAGEDATA", imagedata)
    svg = svg.replace("LABELS", song_data["label"])
    svg = svg.replace("ARTISTS", song_data["artist"])
    svg = svg.replace("DATE", song_data["date"])
    svg = svg.replace("TRACKNAME", song_data["title"])

    for idx in tqdm.tqdm(range(frames), unit=" frames", ncols=80):
        pos = (((idx / rate)) * sr) / stride / bars
        off = int(pos)
        loc = pos - off
        denvs = []
        for env in envs:
            env1 = env[off * bars:(off + 1) * bars]
            env2 = env[(off + 1) * bars:(off + 2) * bars]

            # we want loud parts to be updated faster
            maxvol = math.log10(1e-4 + env2.max()) * 10
            speedup = np.clip(interpole(-6, 0.5, 0, 2, maxvol), 0.5, 2)
            w = sigmoid(speed * speedup * (loc - 0.5))
            denv = (1 - w) * env1 + w * env2
            denv *= smooth
            denvs.append(denv)
        draw_env(denvs, tmp / f"{idx:06d}.png", (fg_color, fg_color2), bg_color, size, svg)

    audio_cmd = []
    if seek is not None:
        audio_cmd += ["-ss", str(seek)]
    audio_cmd += ["-i", audio.resolve()]
    if duration is not None:
        audio_cmd += ["-t", str(duration)]
    print("Encoding the animation video... ")
    # https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
    sp.run([
        "ffmpeg", "-y", "-loglevel", "panic", "-r",
        str(rate), "-f", "image2", "-s", f"{size[0]}x{size[1]}", "-i", "%06d.png"
    ] + audio_cmd + [
        "-c:a", "aac", "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-shortest",
        out.resolve()
    ],
        check=True,
        cwd=tmp)


def parse_color(colorstr):
    """
    Given a comma separated rgb(a) colors, returns a 4-tuple of float.
    """
    try:
        r, g, b = [float(i) for i in colorstr.split(",")]
        return r, g, b
    except ValueError:
        fatal("Format for color is 3 floats separated by commas 0.xx,0.xx,0.xx, rgb order")
        raise


def main():
    parser = argparse.ArgumentParser(
        'seewav', description="Generate a nice mp4 animation from an audio file.")
    parser.add_argument("-r", "--rate", type=int, default=30, help="Video framerate.")
    parser.add_argument("--stereo", action='store_true',
                        help="Create 2 waveforms for stereo files.")
    parser.add_argument("-c",
                        "--color",
                        default=[1, 1, 1],
                        type=parse_color,
                        dest="color",
                        help="Color of the bars as `r,g,b` in [0, 1].")
    parser.add_argument("-c2",
                        "--color2",
                        default=[0.5, 0.3, 0.6],
                        type=parse_color,
                        dest="color2",
                        help="Color of the second waveform as `r,g,b` in [0, 1] (for stereo).")
    parser.add_argument("--white", action="store_true",
                        help="Use white background. Default is black.")
    parser.add_argument("-B",
                        "--bars",
                        type=int,
                        default=80,
                        help="Number of bars on the video at once")
    parser.add_argument("-O", "--oversample", type=float, default=4,
                        help="Lower values will feel less reactive.")
    parser.add_argument("-T", "--time", type=float, default=0.4,
                        help="Amount of audio shown at once on a frame.")
    parser.add_argument("-S", "--speed", type=float, default=4,
                        help="Higher values means faster transitions between frames.")
    parser.add_argument("-W",
                        "--width",
                        type=int,
                        default=1920,
                        help="width in pixels of the animation")
    parser.add_argument("-H",
                        "--height",
                        type=int,
                        default=1080,
                        help="height in pixels of the animation")
    parser.add_argument("-C",
                        "--cover",
                        type=Path,
                        default="cover.png",
                        help="cover path")
    parser.add_argument("-s", "--seek", type=float, help="Seek to time in seconds in video.")
    parser.add_argument("-d", "--duration", type=float, help="Duration in seconds from seek time.")
    parser.add_argument("audio", type=Path, help='Path to audio file')
    parser.add_argument("out",
                        type=Path,
                        nargs='?',
                        default=Path('out.mp4'),
                        help='Path to output file. Default is ./out.mp4')
    parser.add_argument("--artist",
                        nargs='?',
                        default="artist",
                        help='Song artist(s)')
    parser.add_argument("--label",
                        nargs='?',
                        default="label",
                        help='Song label(s)')
    parser.add_argument("--date",
                        nargs='?',
                        default="date",
                        help='Song release date')
    parser.add_argument("--title",
                        nargs='?',
                        default="title",
                        help='Song title')
    args = parser.parse_args()
    songData = {
        "title": args.title,
        "artist": args.artist,
        "date": args.date,
        "label": args.label
    }
    with tempfile.TemporaryDirectory() as tmp:
        visualize(args.audio,
                  Path(tmp),
                  args.out,
                  seek=args.seek,
                  duration=args.duration,
                  rate=args.rate,
                  bars=args.bars,
                  speed=args.speed,
                  oversample=args.oversample,
                  time=args.time,
                  fg_color=args.color,
                  fg_color2=args.color2,
                  bg_color=[1. * bool(args.white)] * 3,
                  size=(args.width, args.height),
                  stereo=args.stereo,
                  cover_path=args.cover,
                  song_data=songData)


if __name__ == "__main__":
    _is_main = True
    main()
