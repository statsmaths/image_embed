# -*- coding: utf-8 -*-
import click
import os
import numpy as np
from glob import glob
import pandas as pd
import umap

@click.command()
# @click.argument('files', nargs=-1, type=click.Path())
@click.option('--model-name', default="mobilenet",
              help='Name of the neural network to load.')
@click.option('--depth', default=-2)
@click.option('--output', default="image-embed.csv")
def main(model_name, depth, output):
    from image_embed import ImageEmbedder

    if os.path.exists(output):
        msg = "Output location '{0:s}' already exists.".format(output)
        raise ValueError(msg)

    input_dir = "~/local/cv_tutorial/images/wikiart"
    input_dir = os.path.expanduser(input_dir)
    input_dir = os.path.abspath(input_dir)

    fs = [y for x in os.walk(input_dir) for y in glob(os.path.join(x[0], '*.jpg'))]

    ie = ImageEmbedder()
    ie.load_model(model_name=model_name, depth=depth)

    ld = len([x.name for x in ie.nn.layers if x.count_params() > 0])
    sz = np.prod(ie.nn.layers[-1].output_shape[1:])

    click.echo("")
    click.echo("Loaded model.       : {0:s}".format(model_name))
    click.echo("Depth parameters    : {0:d}".format(depth))
    click.echo("Total depth         : {0:d}".format(ld))
    click.echo("Output size         : {0:d}".format(sz))
    click.echo("")
    click.echo("Input location      : {0:s}".format(input_dir))
    click.echo("Num. input images   : {0:d}".format(len(fs)))
    click.echo("")

    embed = ie.process_images(fs)

    ue = umap.UMAP(n_neighbors=10,
                   min_dist=0.1).fit_transform(embed)

    df = pd.DataFrame(ue)
    df.columns = ['umap{0:d}'.format(x) for x in range(df.shape[1])]
    df = pd.concat([pd.DataFrame({"file": fs}), df], axis=1)
    df.to_csv(output, index=False)

    click.echo("\n\n")



if __name__ == "__main__":
    main()