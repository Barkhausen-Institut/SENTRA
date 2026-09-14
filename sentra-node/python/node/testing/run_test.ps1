$job1 = Start-Job -ScriptBlock { C:\Users\BhavishMohee\AppData\Local\Microsoft\WindowsApps\python.exe run_mnist_batched_secure.py --node-id 1 --n-nodes 3 --base-port 9200 --host localhost --batch-size 64 --num-epochs 1 --learning-rate 0.05 --t 1 --mnist-samples 1000 --enable-network > node_1_final.log 2>&1 }

$job2 = Start-Job -ScriptBlock { C:\Users\BhavishMohee\AppData\Local\Microsoft\WindowsApps\python.exe run_mnist_batched_secure.py --node-id 2 --n-nodes 3 --base-port 9200 --host localhost --batch-size 64 --num-epochs 1 --learning-rate 0.05 --t 1 --mnist-samples 1000 --enable-network > node_2_final.log 2>&1 }

$job3 = Start-Job -ScriptBlock { C:\Users\BhavishMohee\AppData\Local\Microsoft\WindowsApps\python.exe run_mnist_batched_secure.py --node-id 3 --n-nodes 3 --base-port 9200 --host localhost --batch-size 64 --num-epochs 1 --learning-rate 0.05 --t 1 --mnist-samples 1000 --enable-network > node_3_final.log 2>&1 }

Wait-Job $job1, $job2, $job3
