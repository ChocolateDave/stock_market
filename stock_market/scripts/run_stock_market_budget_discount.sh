SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Running Stock Market MADDPG with different Learning Rates";
python ${SCRIPT_DIR}/../train.py --num-agents 5 -bs 256 --gpu --gpu-id $1 --budget-discount 0.9 --exp-name stock_market_10_bs_budget_discount_0.9 & 
python ${SCRIPT_DIR}/../train.py --num-agents 5 -bs 256 --gpu --gpu-id $1 --budget-discount 0.95 --exp-name stock_market_10_bs_256_budget_discount_0.95 &
python ${SCRIPT_DIR}/../train.py --num-agents 5 -bs 256 --gpu --gpu-id $1 --budget-discount 0.99 --exp-name stock_market_10_bs_256_budget_discount_0.99 &
wait
echo "All Done!"