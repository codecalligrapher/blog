---
title: "Zero-Padding a CSV with AWK"
date: 2022-08-26T00:00:00-00:00
math: true
cover: 
    image: "blog/covers/awk_zero_pad.png"
    relative: true
---
This was purely out of sheer need, and this was the fastest way I could've gotten it done (I ended up learning a LOT about CLI and the `awk` command from this, so I'm really grateful for that)

The problem: I have a column in a `utf-8` CSV file of type Integer, which should actually be type string and zero-padded up to (let's say length N).

```bash
~/projects/awk_pad ❯ cat out.csv             
a,Y,1
b,N,10
c,Y,12223253
```

What I want, is the following (output from the `cat` tool):

```bash
~/projects/awk_pad ❯ cat out_clean.csv
a,Y,00000000001
b,N,00000000010
c,Y,00012223253
```


In order to accomplish the above, the following was used:

```bash
~/p/awk_pad ❯ awk -F ',' '{OFS=","};{print $1, $2, sprintf("%011d", $3)}' out.csv > out_clean.csv
```

Let's break this down, the `awk` command is a typical *NIX tool, which according to the manual page is short for `gawk`, a "pattern scanning and processing language". I'm not going into the interals, since it's far too detailed; instead, I'd cover the above command alone.

The general syntax is as follows:  
```
awk options 'selection _criteria {action }' input-file > output-file
```

Firstly, `-F ','` is an option used to tell `awk` that the input file is comma-separated. This allows accesses of line-elements by the `$` (i.e., `$2` would access the 2nd element of every line), since `awk` works on a line-by-line basis. 

The first part of the of the *action* is `{OFS=","}`, which tells `awk` that all arguments must be separated by the comma in output. 

The second part of the *action* is `{print $1, $2, sprintf("%011d", $3)}`. This tells awk to output the first and second arguments (think first and second columns of the CSV file), followed by a zero-padded version of the third argument (column). 

The `%011d` in this case says "print with precision of 11", which ensures that the outpu is ALWAYS length eleven, anf if not is instead zero-padded. If the zero were replaced by a blank space " ", the resulting would have been space-padded strings.
