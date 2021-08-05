printf "\n[INFO] Running test images 0-15 (this may take a while)...\n"

python -m darts clean
python -m darts dart0.jpg
python -m darts dart1.jpg
python -m darts dart2.jpg
python -m darts dart3.jpg
python -m darts dart4.jpg
python -m darts dart5.jpg
python -m darts dart6.jpg
python -m darts dart7.jpg
python -m darts dart8.jpg
python -m darts dart9.jpg
python -m darts dart10.jpg
python -m darts dart11.jpg
python -m darts dart12.jpg
python -m darts dart13.jpg
python -m darts dart14.jpg
python -m darts dart15.jpg

printf "\n[INFO]: Testing complete! (see /darts/out)\n\n"
