for i in /home/matthew/Programs/scribe/projects/tacotron/trainer/dataYaleYoutube/raw/*.txt; do 
  echo "\"wavs/$(basename $i| sed 's/txt/wav/g')\":[\"$(cat $i)\"]," >> alignment.json
done
