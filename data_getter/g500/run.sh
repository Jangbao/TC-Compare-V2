export G500_PATH=$(pwd)/cache.bin
cd graph500-2.1.4
make
cd ..

cp ./graph500-2.1.4/seq-csr/seq-csr seq-csr
chmod +x seq-csr 



./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e2/edges.bin  -R  -s 17  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e8/edges.bin  -R  -s 17  -e 8  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e32/edges.bin  -R  -s 17  -e 32  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e128/edges.bin  -R  -s 17  -e 128  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s19-e2/edges.bin  -R  -s 19  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s19-e8/edges.bin  -R  -s 19  -e 8  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s19-e32/edges.bin  -R  -s 19  -e 32  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s19-e128/edges.bin  -R  -s 19  -e 128  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s21-e2/edges.bin  -R  -s 21  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s21-e8/edges.bin  -R  -s 21  -e 8  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s21-e32/edges.bin  -R  -s 21  -e 32  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s21-e128/edges.bin  -R  -s 21  -e 128  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s23-e2/edges.bin  -R  -s 23  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s23-e8/edges.bin  -R  -s 23  -e 8  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s23-e32/edges.bin  -R  -s 23  -e 32  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s23-e128/edges.bin  -R  -s 23  -e 128  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  

./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e2/edges.bin  -R  -s 17  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e8/edges.bin  -R  -s 17  -e 8  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e32/edges.bin  -R  -s 17  -e 32  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e128/edges.bin  -R  -s 17  -e 128  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s19-e2/edges.bin  -R  -s 19  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s19-e8/edges.bin  -R  -s 19  -e 8  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s19-e32/edges.bin  -R  -s 19  -e 32  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s19-e128/edges.bin  -R  -s 19  -e 128  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s21-e2/edges.bin  -R  -s 21  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s21-e8/edges.bin  -R  -s 21  -e 8  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s21-e32/edges.bin  -R  -s 21  -e 32  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s21-e128/edges.bin  -R  -s 21  -e 128  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s23-e2/edges.bin  -R  -s 23  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s23-e8/edges.bin  -R  -s 23  -e 8  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s23-e32/edges.bin  -R  -s 23  -e 32  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s23-e128/edges.bin  -R  -s 23  -e 128  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  

./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s18-e2/edges.bin  -R  -s 18  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s20-e2/edges.bin  -R  -s 20  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s22-e2/edges.bin  -R  -s 22  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s24-e2/edges.bin  -R  -s 24  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s26-e2/edges.bin  -R  -s 26  -e 2  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e4/edges.bin  -R  -s 17  -e 4  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e16/edges.bin  -R  -s 17  -e 16  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e64/edges.bin  -R  -s 17  -e 64  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e256/edges.bin  -R  -s 17  -e 256  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster0-s17-e1024/edges.bin  -R  -s 17  -e 1024  -A 0.25 -B 0.25  -C 0.25 -D 0.25 &  

./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s18-e2/edges.bin  -R  -s 18  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s20-e2/edges.bin  -R  -s 20  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s22-e2/edges.bin  -R  -s 22  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s24-e2/edges.bin  -R  -s 24  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s26-e2/edges.bin  -R  -s 26  -e 2  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e4/edges.bin  -R  -s 17  -e 4  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e16/edges.bin  -R  -s 17  -e 16  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e64/edges.bin  -R  -s 17  -e 64  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e256/edges.bin  -R  -s 17  -e 256  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  
./seq-csr  /home/jiangbo/TC-compare-clean/data/g500_dataset/cluster3-s17-e1024/edges.bin  -R  -s 17  -e 1024  -A 0.75 -B 0.1  -C 0.1 -D 0.05 &  


./seq-csr  /home/LiJB/cuda_project/TC-compare-V100/dataset_app/g500_dataset/cluster2-s20-e512/edges.bin  -R  -s 20  -e 512