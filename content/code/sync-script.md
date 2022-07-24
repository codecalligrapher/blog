---
title: "Sync Script"
date: 2022-07-24T12:38:42-04:00
ShowCodeCopyButtons: true
hideSummary: true
keywords: ["bash", "linux"]
comments: true
showReadingTime: false
---

```bash
#!/usr/bin/zsh

while getopts a: flag
do
  case "${flag}" in
    a) remote=${OPTARG};;
  esac
done

echo "remote IP: $remote";
echo "write: $final";

home_dir="/home/aadi/"
storage_dir="/storage/"


# files in home
HomeArray=(
  ".zshrc"
)

if [[ $* == *--write* ]] 
then
  for arr in $HomeArray; do
    rsync -Pvtau $home_dir$arr aadi@$remote:$home_dir$arr
  done
else
  for arr in $HomeArray; do
    rsync -Pvntau $home_dir$arr aadi@$remote:$home_dir$arr
  done
fi

# folders in home
HomeArray=(
  ".config/alacritty"
  ".config/qtile"
  ".config/nvim"
)

if [[ $* == *--write* ]] 
then
  for arr in $HomeArray; do
    rsync -Pvtau $home_dir$arr/ aadi@$remote:$home_dir$arr
  done
else
  for arr in $HomeArray; do
    rsync -Pvntau $home_dir$arr/ aadi@$remote:$home_dir$arr
  done
fi

# files in storage
SArray=(
  "reading"
  "research"
)

if [[ $* == *--write* ]] 
then
  for arr in $SArray; do
    rsync -Pvtau $storage_dir$arr/ aadi@$remote:$home_dir$arr
  done
else
  for arr in $SArray; do
    rsync -Pvntau $storage_dir$arr/ aadi@$remote:$home_dir$arr
  done
fi
```
