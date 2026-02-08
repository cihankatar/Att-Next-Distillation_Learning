#!/bin/bash

#SBATCH -p barbun-cuda      							# Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ckatar          							# Kullanici adi
#SBATCH -J Dinobarbuntraining  								# Gonderilen isin ismi
#SBATCH -o dinobarbun.out			  			        	# Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        							# Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                							# Gorev kac node'da calisacak?
#SBATCH -n 1                							# Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20							# Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=20:00:00    							# Sure siniri koyun. (HH:MM:SS)
#SBATCH --mail-type=begin        						# send email when job begins
#SBATCH --mail-type=end          						# send email when job ends
#SBATCH --mail-user=cihan.katar@tau.edu.tr

module load miniconda3
conda activate segmentation
cd /arf/home/ckatar/Desktop/Repo/Att-Next-Distillation_Learning
git commit --amend
python train_dino.py