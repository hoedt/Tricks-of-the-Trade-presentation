# Neural Networks: Tricks of the Trade

These are the resources I used for my presentation on tricks of the trade for the AIDD workshop in 2021.
Normally, I create my presentations using libreoffice or google slides, but this time I wanted to give [revealjs](https://revealjs.com) a try.
Since I have not yet completely figured things out yet, there might be some suboptimal solutions in there.
Feel free to let me know if you know of improvements (e.g., how to improve citing/references).

 - A recording of this presentation is available on [vimeo](https://vimeo.com/637888422).
 - The slides can be directly viewed in the browser [here](https://hoedt.github.io/Tricks-of-the-Trade-presentation/).

### Generated Images

The custom images for this presentation were generated in a python environment with following packages:

```plain
numpy~=1.19.2
matplotlib~=3.3.4
pytorch~=1.8.0
torchvision~=0.9.0
```

Simply running the script in the `resources` directory should generate the images:

```bash
cd resources
python normalisation.py
cd ..
```

### Further Reading

My main resources are linked in the presentation, but probably there is a lot more information out there.
If you know of articles/blog posts that share useful tricks for deep learning,
then let me know so I can add them to the list below:

 - [Neural Networks: Tricks of the Trade](https://link.springer.com/book/10.1007/978-3-642-35289-8) (2nd edition)
 - [A recipe for training neural networks](http://karpathy.github.io/2019/04/25/recipe/) (DL in practice)
 - [The Sorcerer’s Apprentice Guide to Training LSTMs](https://www.niklasschmidinger.com/posts/2020-09-09-lstm-tricks/) (LSTM tricks)
 - ...
