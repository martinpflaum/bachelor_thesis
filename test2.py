import click
@click.command()
@click.option('--n', default=1)
def dots(**config_kwargs):
    print(type(config_kwargs))
    #click.echo('.' * n)


if __name__ == '__main__':
    dots()