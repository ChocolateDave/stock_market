SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Running Stock Market MADDPG with different Learning Rates";
python ${SCRIPT_DIR}/../train.py --num-agents 10 -bs 256 --gpu --gpu-id $1 -lr 0.0001 --exp-name stock_market_10_bs_lr_0.0001 &
python ${SCRIPT_DIR}/../train.py --num-agents 10 -bs 256 --gpu --gpu-id $1 -lr 0.001 --exp-name stock_market_10_bs_256_lr_0.001 &
python ${SCRIPT_DIR}/../train.py --num-agents 10 -bs 256 --gpu --gpu-id $1 -lr 0.01 --exp-name stock_market_10_bs_256_lr_0.01 &
wait
echo "All Done!"