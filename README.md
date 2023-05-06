# PROTEIN GENERATOR: Generate sequence-structure pairs with RoseTTAFold
<img src='http://files.ipd.uw.edu/pub/sequence_diffusion/figs/seqdiff_anim_720p.gif' width='600' style="vertical-align:middle">

![gif](http://files.ipd.uw.edu/pub/sequence_diffusion/figs/seqdiff_anim_720p.gif)

## Getting Started
The easiest way to get started is with [PROTEIN GENERATOR](https://huggingface.co/spaces/merle/PROTEIN_GENERATOR) a HuggingFace space where you can play around with the model!

Before running inference you will need to set up a custom conda environment (note if you are running this on the digs cluster SE3nv environment can be used).

Start by creating a new environment <code>conda create --name MYENV</code> and activating it <code>source activate MYENV</code>

Next pip install the requirements file <code>pip install -r requirements.txt</code> (note depending on what type of GPU you are on the dgl package required may change see [here](https://www.dgl.ai/pages/start.html))

Once everything has been installed you can download checkpoints:
- [base checkpoint](http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_221219_equalTASKS_nostrSELFCOND_mod30.pt)
- [DSSP + hotspot checkpoint](http://files.ipd.uw.edu/pub/sequence_diffusion/checkpoints/SEQDIFF_230205_dssp_hotspots_25mask_EQtasks_mod30.pt)

You will have to add the checkpoints at the top of the [sampler class](utils/sampler.py)

The easiest way to get started is opening the <code>protein_generator.ipynb</code> notebook and running the sampler class interactively, when ready to submit a production run use the output <code>agrs.json</code> file to launch: 

<code>python ./inference.py -input_json ./examples/out/design_000000_args.json</code> 


Check out the templates in the [example folder](examples) to see how you can set up jobs for the various design strategies


## Adding new sequence based potentials
To add a custom potential to guide the sequence diffusion process toward your desired space, you can add potentials into <code>utils/potentials.py</code>. At the top of the file a template class is provided with functions that are required to implement your potential. It can be helpful to look through the other potentials in this file to see examples of how to implement. At the bottom of the file is a dictionary mapping the name used in the <code>potentials</code> argument to the class name in file. 

![pic](http://files.ipd.uw.edu/pub/sequence_diffusion/figs/diffusion_landscape.png)

## About the model
PROTEIN GENERATOR is trained on the same dataset and uses the same architecture as RoseTTAFold. To train the model, a ground truth sequence is transformed into an Lx20 continuous space and gaussian noise is added to diffuse the sequence to the sampled timestep. To condition on structure and sequence, the structre for a motif is given and then corresponding sequence is denoised in the input. The rest of the structure is blackhole initialized. For each example the model is trained to predict Xo and losses are applied on the structure and sequence respectively. During training big T is set to 1000 steps, and a square root schedule is used to add noise.


## Looking ahead
We are interested in problems where diffusing in sequence space is useful, if you would like to chat more or join in our effort for sequence diffusion come talk to Sidney or Jake!

## Acknowledgements
A project by Sidney Lisanza and Jake Gershon. Thanks to Sam Tipps for implementing symmetric sequence diffusion. Thank you to Minkyung Baek and Frank Dimaio for developing RoseTTAFold, Joe Watson and David Juergens for the developing inpainting inference script which the inference code is built on top of.

