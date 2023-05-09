# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line for audio compression."""
import argparse
from pathlib import Path
import sys
import torchaudio
import os
from net3 import SoundStream
import torch
import typing as tp
from collections import OrderedDict
import librosa
SUFFIX = '.ecdc'
def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def get_parser():
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    parser.add_argument(
        'input', type=Path,
        help='Input file, whatever is supported by torchaudio on your system.')
    parser.add_argument(
        'output', type=Path, nargs='?',
        help='Output file, otherwise inferred from input file.')
    parser.add_argument('--resume_path', type=str, default='resume_path', 
                        help='resume_path')
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='Automatically rescale the output to avoid clipping.')
    return parser


def fatal(*args):
    print(*args, file=sys.stderr)
    sys.exit(1)


def check_output_exists(args):
    if not args.output.parent.exists():
        fatal(f"Output folder for {args.output} does not exist.")
    if args.output.exists() and not args.force:
        fatal(f"Output file {args.output} exist. Use -f / --force to overwrite.")


def check_clipping(wav, args):
    if args.rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


def check_clipping2(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


def test_one(wav_root, store_root, rescale, args, soundstream):
    #compressing
    wav, sr = torchaudio.load(wav_root)
    #wav = wav*0.95
    # wav = librosa.core.load(wav_root, sr=24000)[0]
    # wav = torch.from_numpy(wav).unsqueeze(0)
    wav = wav.unsqueeze(1).cuda()
    #print('wav ', wav.shape)
    compressed = soundstream.encode(wav, target_bw=12)
    #print('compressed ', compressed) # (n_q, B, len)
    
    # assert 1==2
    #print(wav_root)
    #print('finish compressing')
    out = soundstream.decode(compressed)
    # print('out ', out.shape)
    # assert 1==2
    out = out.detach().cpu().squeeze(0)
    check_clipping2(out, rescale)
    save_audio(out, store_root, 16000, rescale=rescale)
    print('finish decompressing')
    #assert 1==2

def remove_encodec_weight_norm(model):
    from modules import SConv1d
    from modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)

def test_batch():
    args = get_parser().parse_args()
    if not args.input.exists():
        fatal(f"Input file {args.input} does not exist.")
    input_lists = os.listdir(args.input)
    input_lists.sort()
    soundstream = SoundStream(n_filters=32, D=512, ratios=[8, 5, 4, 2])
    parameter_dict = torch.load(args.resume_path)
    new_state_dict = OrderedDict()
    for k, v in parameter_dict.items(): # k为module.xxx.weight, v为权重
        name = k[7:] # 截取`module.`后面的xxx.weight
        new_state_dict[name] = v
    soundstream.load_state_dict(new_state_dict) # load model
    #remove_encodec_weight_norm(soundstream)
    soundstream = soundstream.cuda()
    os.makedirs(args.output, exist_ok=True)
    for audio in input_lists:
        test_one(os.path.join(args.input,audio), os.path.join(args.output,audio), args.rescale, args, soundstream)

if __name__ == '__main__':
    #main()
    test_batch()