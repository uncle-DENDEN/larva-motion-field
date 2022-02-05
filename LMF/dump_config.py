from pathlib import Path
import os
import yaml
import io


def dump_to_dict(**kwargs):
    return kwargs


data = dump_to_dict(video_name='',  # video name or video folder name
                    root_path=r'D:\workspace\python\objectTracking\sample',
                    out_path=r'D:\workspace\python\objectTracking\out_file',
                    period_range=(457, 461),  # estimated period of recurrent background movement
                    periodic_thresh=1,  # threshold for detecting a pixel as a recurrent pixel
                    kernel_size=(6, 6),  # kernel used for morphological opening and closing
                    beta=1.0,  # pruning accuracy tolerance, 0.0 < beta <= 1.0
                    size_range=(0, 1000),  # size filter
                    # configuration  of the morphotracker
                    length=64,  # length of larva in pixel, must be integer
                    maxDisappeared=270,
                    # maxDisappeared frames of a subjects. If exceeds, then the object is deregistered
                    warmup=10,  # start tracking after the warmup period
                    # configuration  of the analyzer
                    matplotlib_backend='qt5agg',
                    framerate=7)

with io.open('config.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True, sort_keys=False)
