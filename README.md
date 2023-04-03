# Hide_and_Speak_Real_Imag

This is a Speech Steganography Project. In this a message audio(m) is hidden inside a carrier audio (c) using an encoder to get (c_hat) which sounds similar to the original carrier. The hidden message can be recoverd by passing c_hat through the decoder to get m_hat. The recovered message should sound like the original message.

//The baseline is Hide and Speak: Towards Deep Neural Networks for Speech Steganography (https://arxiv.org/abs/1902.03083)

However, the approach we used differs from the baseline in the sense that, the Phase information in the audios are preserved. Each speech spectrogram is seperated into real and imaginary parts and concatenated together which leads to each audio having 2 channels. This can again be used to recover the audios.


The different version are: 

Hide 1 message inside carrier \n
Hide 2 messages inside carrier \ n
Hide 3 messages inside carrier
