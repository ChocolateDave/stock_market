SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Running Stock Market MADDPG with different Learning Rates";
python ${SCRIPT_DIR}/../train.py --num-agents 5 --gpu --gpu-id $1 --budget-discount 0.50 --exp-name stock_market_5_budget_discount_0.50 & 
python ${SCRIPT_DIR}/../train.py --num-agents 5 --gpu --gpu-id $1 --budget-discount 0.90 --exp-name stock_market_5_budget_discount_0.90 &
python ${SCRIPT_DIR}/../train.py --num-agents 5 --gpu --gpu-id $1 --budget-discount 0.95 --exp-name stock_market_5__budget_discount_0.95 &
wait
echo "All Done!"