# Text-to-Image Generation with Transformers and Diffusions
Generative AI is evolving rapidly, revolutionizing every aspect of our lives and work. Text-to-image models, in particular, have gained significant attention due to their ability to translate natural language into visually rich, meaningful images. Models like OpenAI’s DALL-E series, Google’s Imagen, and Stability AI’s Stable Diffusion have shown unprecedented advances in the field of generative AI, turning abstract descriptions into detailed, highly creative visual representations.

I focus on two ways of generating images from text prompts: vision Transformers (ViT) and diffusion models. In the first method, we divide an image into multiple patches and treat each patch as an element in a sequence. As a result, image generation becomes a sequence prediction problem. When generating an image based on a text prompt, we ask the trained Transformer to first predict the top left patch. In the next iteration, we feed the first patch, along with the prompt, to the ViT and ask it to predict the second patch. We repeat the process until we have all the needed patches in the image.

You'll use "panda with top hat reading a book" as the text prompt. You’ll learn to load up the four components of the trained model: the text tokenizer, the BART encoder, the BART decoder, and the VQGAN detokenizer. You'll see how the prompt is converted to tokens, and then indexes, which are used as inputs to the BART encoder. The encoded text is then fed to the BART decoder to predict image tokens in an autoregressive fashion (the top left image patch first, then the one to the right, and so on). The decoded image tokens are fed to the VQGAN detokenizer to convert them into a high-resolution image for you to see on your computer.
The generated image based on the prompt "panda with top hat reading a book" is shown below:

![fig1 1](https://github.com/user-attachments/assets/168bcde4-498b-49a5-a83d-145c408ddc1b)

We also select 8 intermediate steps to visualize how the image was generated step by step, as shown below.

![fig1 2](https://github.com/user-attachments/assets/c302b833-7419-4187-9517-b9d5ba640a77)


Better yet, in an animation, you'll visualize the intermediate steps of the image generation process. What does the image look like when only 25 of the 265 patches are generated? What if 58 patches are generated? and so on. The animation to illustrate the image generation process can be seen on my website here https://gattonweb.uky.edu/faculty/lium/v/minDALLE.gif.

We display 8 images in the 2 by 4 grid. An image is divided into 256 patches, organized in a 16x16 grid. The top left image shows the output when 25 image patches are generated. The second image in the top row shows the output when 58 patches are generated. The rest images show the outputs when 91, 124, ..., and 256 patches are generated.

The second way of text-to-image generation is based on diffusion models. We start with an image with pure noise. We ask the trained diffusion model to denoise it slightly, conditional on the text prompt. The result is a less noisy image. We repeat the process many times until we obtain a clean image that matches the text prompt. Diffusion models have become the go-to generative model by learning progressively removing noise from noisy images. They form the foundation of the state-of-the-art text-to-image models, including Imagen, DALL-E 2 (though not DALL-E), and Stable Diffusion.

When we use the prompt "a banana riding a motorcycle with sunglasses and a straw hat," the generated image is below:

![output_35_0](https://github.com/user-attachments/assets/7fe6ab46-c18a-415a-98ed-9e9adfa7e86b)

Better yet, you can visualize the intermediate outputs from the diffusion model. In the image below, you see the generated image at different time steps during the reverse diffusion process. The left image shows that at time step t=800, the image looks close to pure noise. As we move to the right, the image starts to match the text prompt. At time step t=0, you can see a clean image of a banana riding a motorcycle. 

![output_39_0](https://github.com/user-attachments/assets/94f27794-f5c3-45e5-8f59-18c2356057b4)

Diffusion models are also excellent image editors. In the figure below, the original horse image is in the middle. When we use the prompt "white horse on sunny beach" to edit the image, the result is shown on the left of the figure. When we use the prompt "zebra on snowy ground" to edit the image, the result is shown on the right of the figure.


![edit](https://github.com/user-attachments/assets/d2569f05-992c-4cf9-bcd6-a3e5a1f9e39d)

