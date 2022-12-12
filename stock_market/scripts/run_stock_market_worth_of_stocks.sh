SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Running Stock Market MADDPG with different Learning Rates";
python ${SCRIPT_DIR}/../train.py --num-agents 5 --gpu --gpu-id $1 --worth-of-stocks 0.1 --exp-name stock_market_5_worth_of_stock_0.10 & 
python ${SCRIPT_DIR}/../train.py --num-agents 5 --gpu --gpu-id $1 --worth-of-stocks 0.5 --exp-name stock_market_5_worth_of_stock_0.50 &
python ${SCRIPT_DIR}/../train.py --num-agents 5 --gpu --gpu-id $1 --worth-of-stocks 0.9 --exp-name stock_market_5_worth_of_stock_0.90 &
wait
echo "All Done!"