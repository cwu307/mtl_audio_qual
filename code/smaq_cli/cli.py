#!/usr/bin/env python
"""
Welcome to SMAQ predictor python Command Line Interface (CLI)

This CLI tool allows you to assess the perceptual quality of a compressed (degraded) audio file given it's
uncompressed counterpart. The returned SMAQ score is a number from 1.0 to 5.0; their semantic meanings are:
5: Excellent
4: Good
3: Fair
2: Poor
1: Bad

Additionally, the CLI tool returns the raw scores and raw audio features which helped the model arrived at the
final score.
================================================================
Example Usage:
>> (.venv)$ python cli.py -t <target_filepath> -r <reference_filepath>
>>
SMAQ score = 3.0711207389831543
raw score = [[6.8538374e-01 4.2283174e-01 4.2692372e-01 5.8093834e-01 2.1576850e-06
1.8153571e-02]]
raw features = [ 0.84460003 -8.63695857  0.86474614  0.77688329]
================================================================
"""

import click
import runez

from smaq_cli.smaq_predictor import SmaqPredictor


@runez.click.command()
@runez.click.version()
@click.option("-r", "--ref_filepath", required=True)
@click.option("-t", "--tar_filepath", required=True)
@click.option("-m", "--model_filepath", help="[optional] filepath of a pre-trained model", required=False)
@click.option("-s", "--scaler_filepath", help="[optional] filepath of a pre-trained feature scaler", required=False)
@click.option("-o", "--output_json_filepath", help="[optional] filepath of the output json file", required=False)
@click.option("--debug", is_flag=True, help="Show debugging information.")
@runez.click.log()
def main(ref_filepath, tar_filepath, model_filepath, scaler_filepath, output_json_filepath, debug, log):
    """
    Description of command (shows up in --help)
    """
    runez.log.setup(debug=debug, file_location=log, locations=None, greetings=":: {argv}")
    print("Processing...\ntarget = %s\nreference = %s" % (tar_filepath, ref_filepath))
    print("=============================================================")
    smaq_predictor = SmaqPredictor()
    # ==== set model and scaler path if provided
    if model_filepath and scaler_filepath:
        print("model filepath = %s" % model_filepath)
        print("scaler filepath = %s" % scaler_filepath)
        smaq_predictor.set_model_scaler_path(model_filepath, scaler_filepath)
    # TODO: further optimization might be needed for determining the best segment size
    # ==== invoke predict() and get SMAQ score!
    final_score, raw_score, features = smaq_predictor.predict(tar_filepath, ref_filepath)
    print("=============================================================")
    print("SMAQ score = %s" % final_score)
    print("raw score = %s" % raw_score)
    print("raw features = %s" % features)
    print("=============================================================")
    # ==== write to JSON file
    if output_json_filepath:
        print("write results to %s" % output_json_filepath)
        smaq_predictor.write_json_output(output_json_filepath)


if __name__ == "__main__":  # pragma: no cover
    # This is not needed with click, only there for convenience, if one wants to directly invoke this script
    main()
