# Hide_and_Speak_Real_Imag

This is a Speech Steganography Project. In this a message audio(m) is hidden inside a carrier audio (c) using an encoder to get (c_hat) which sounds similar to the original carrier. The hidden message can be recoverd by passing c_hat through the decoder to get m_hat. The recovered message should sound like the original message.


However, the approach we used differs from the baseline in the sense that, the Phase information in the audios are preserved. Each speech spectrogram is seperated into real and imaginary parts and concatenated together which leads to each audio having 2 channels. This can again be used to recover the audios.


The different version are: 

Hide 1 message inside carrier   
Hide 2 messages inside carrier  
Hide 3 messages inside carrier  
Hide 2 messages inside a carrier and a conditionl decoder to decode the message using a key

(All model checkpoints are available)

To run the code:  
(You can change the hyperparameters accordingly)


**Training**  
python main.py --num_iters 60 --mode train --train_path datasets/Train --val_path datasets/Val --test_path datasets/Test --batch_size 32 --n_pairs 3000 --n_messages 1 --dataset timit --run_dir 1_msg --save_model_every 5 --num_workers 4

**Testing**. 
python main.py --mode test --load_ckpt ckpt/60_epoch --train_path datasets/Train --val_path datasets/Val --test_path datasets/Test  --dataset timit --n_messages 1


**Inference**. 
python main.py --mode sample --run_dirsample --load_ckpt ckpt/60_epoch --train_path datasets/Train --val_path datasets/Val --test_path datasets/Test  --dataset timit --n_messages 1


