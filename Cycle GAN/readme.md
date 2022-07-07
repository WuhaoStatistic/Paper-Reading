# The code is from [Cycle-GAN Github rep](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Loss function

I think understanding the struct of loss function and how it works can help us clearly understand this model.   
Before we start this paret, we should know that we willhave two dataset **domain A** and **domain B**. And we also have two generators and discriminators.  
|Generators| $G_A$ : domain A -> B | $G_B$: domain B->A|
|:--------:|:--------------------:|:----------------:|
|Discriminators| $D_A$ : $G_A(A)$ vs. B | $D_B$ : $G_B(B)$ vs. A|

Basically, We have two loss, one for training generator one for training discriminator.
## $Loss_D$
![image](https://user-images.githubusercontent.com/89610539/177794231-e39df55a-95e0-4adb-9d76-0b79ca7fdbc4.png)

**criterionGAN** is a class for computing GAN loss. The second parameter is a flag to tell the computing function whether input is fake image or real image. Only when the net
knows which one is real and fake can it calculate the difference between current situation and ideal situation.

## $Loss_G$

As mentioned in the paper, we have three components for this parts(**identity loss** is not in original paper but shows up in author's code).

### Gan Loss

```
# GAN loss D_A(G_A(A))
self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

# GAN loss D_B(G_B(B))
self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
```
### Cycle Loss
```
#if G_A(A) can generate 'real_B' and G_B(B) can generate real_Athen this loss should be 0

# Forward cycle loss || G_B(G_A(A)) - A|| 
self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

# Backward cycle loss || G_A(G_B(B)) - B||
self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
```
### Identity Loss
```
# G_A should be identity if real_B is fed: ||G_A(B) - B||
self.idt_A = self.netG_A(self.real_B)
self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

# G_B should be identity if real_A is fed: ||G_B(A) - A||
self.idt_B = self.netG_B(self.real_A)
self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
```
In short way, **Cycle Loss** and **Idt Loss** can help generator control the shape keeps unchanged during genrating.

## Training tricks

### change the updating frequency

Sometimes we may found generator is outperforming than discriminator sometimes not. When this situation happens, we may need to train the weaker one more than another.
In **cycle_gan_model.py**, we can find the function `optimize_parameters()`. By adding a flag, we can easily control the frequency.

### unbalanced weights

For example, my project aims to transfer picture from domain B to domain A. When using this module, I can put more weights on `loss_G_B` and `loss_D_A`(in the paper, $D_Y$ is associated with **real_Y** and **fake_Y** (aka. '**G**(x)'). But here $D_A$ is working with **real_B** and **fake_B**. But it does not matter since it’s only naming problems.)
In **cycle_gan_model.py**,we can find backwords function for G and D, one can modify according to needs.

We can also try modify weights on `cycle_loss` and `idt_loss`. Things are different from datasets to datasets.

### checkerboard effect

One annoying thing here is checkerboard effect. Just like its name, if you dive into every pixel, you may find pixels keep changing from dark-light-dark......

The problems happens in upsampling stage.
The generator in this github can be resnet and U-net. In the resnet, they use transposedconv with
`Kernel_size =3` and `stripe = 2`. In the transposed convolution, once kernel size can not be divided by stripe , the feature map will suffer pixel overlap. So, it’s better to use U-net(`kernel_size = 4` and `stripe = 2`) or we use interpolation+conv instead of transposedconv.

### choice of optimizer

Cycle gan is really difficult to converge(especially when identity loss is added). So, we may use Adam first to meet quick converge. However, when the curve is converge,
we can change to SGD to continue training. BTW,it's mentioned by the author that we can not really tell if module is converged by loss curve, it's better to generate picture
periodically to see the reults.

### dataset

Cycle gan can not do shape transfer.So, it's better if objects in two domains can have similar feature.For example, landmarks in painting and photo is a good choice, whereas
dogs in photo and trees in photo is not that good for this module.

It's also important if we can have picture in high resolution. The **discriminator** here is **patch discriminator**. Like the picture below, discriminator will work
on the green feature map.In the green feature map, one dark green block is the mixture of 9 dark blue blocks in the blue feature map. 
![image](https://user-images.githubusercontent.com/89610539/177802451-975452ac-519c-4da4-b6ff-23e856ea69ce.png)  
This `1-to-n` is where patch comes from. So, if the picture is too small, the higher feature map may mixture features which shouldn't be mixed together.  
![image](https://user-images.githubusercontent.com/89610539/177803546-2d57ae3c-1868-44b3-a059-a7d39a7001b9.png)  
Just like th picture above, on the left side, the trees maybe mixed with sky is the picture is small, and is apprently not good for generating a sky in new picture. Although
higher resolution cannot perfectly solve this problems but it can help on some level. The alternative way is reduce layers in discriminators, but we all know that it will
reduce the capacity of module(after all, by default we only have 3 conv layers in discriminator)


