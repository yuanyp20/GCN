mkdir train_1
cd train_1
mkdir compare
mkdir save_path
mkdir writer
cd ../

for((i=1;i<=10;i++));
do
python train_1.py $i
done


