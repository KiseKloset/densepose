configs="configs/r101_fpn_dl.yaml"
ckpt="ckpt/final.pkl"
input="*input/*.jpg"
output="output"
temp_output="output.pkl"
gpu="cuda:0"

# extract data
python apply_net.py dump $configs $ckpt $input --output $temp_output --opts MODEL.DEVICE $gpu
python parse_densepose.py --out output
rm $temp_output

# visualize
# python apply_net.py show $configs $ckpt $input bbox,dp_segm -v