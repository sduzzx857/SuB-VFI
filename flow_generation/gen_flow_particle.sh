#!/bin/bash
python flow_generation/gen_flow_particle.py --dataset MICROTUBULE --density high
python flow_generation/gen_flow_particle.py --dataset MICROTUBULE --density low
python flow_generation/gen_flow_particle.py --dataset RECEPTOR --density high
python flow_generation/gen_flow_particle.py --dataset RECEPTOR --density low
python flow_generation/gen_flow_particle.py --dataset VESICLE --density high
python flow_generation/gen_flow_particle.py --dataset VESICLE --density low
python flow_generation/gen_flow_particle.py --dataset EB1
python flow_generation/gen_flow_particle.py --dataset CCR5
python flow_generation/gen_flow_particle.py --dataset LYSOSOME