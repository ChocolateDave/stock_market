echo "Running Stock Market MADDPG with different Learning Rates";
python src/train.py --num-agents 10 -bs 256 --gpu --gpu-id $1 -lr 0.0001 --exp-name stock_market_lr_0.0001 &
python src/train.py --num-agents 10 -bs 256 --gpu --gpu-id $1 -lr 0.001 --exp-name stock_market_lr_0.001 &
python src/train.py --num-agents 10 -bs 256 --gpu --gpu-id $1 -lr 0.01 --exp-name stock_market_lr_0.01 &
wait
echo "All Done!"