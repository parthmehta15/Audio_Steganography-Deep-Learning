from loguru import logger
from os.path import join, basename
from hparams import *
from stft.stft import STFT
from dataloader import spect_loader
import soundfile as sf
# from griffin_lim import griffin_lim

def convert(solver, carrier_wav_path, msg_wav_paths, trg_dir, epoch, trim_start, num_samples=16000):
    if solver.mode != 'test':
        logger.warning("generating audio not in test mode!")

    data, sr = sf.read(carrier_wav_path)
    print('Data: ',data.shape)
    carrier_basename = basename(carrier_wav_path).split(".")[0]
    msg_basenames = [basename(msg_wav_path).split(".")[0] for msg_wav_path in msg_wav_paths]

    carrier = spect_loader(carrier_wav_path, trim_start, num_samples=num_samples)
    print('Carrier: ',carrier.shape)
    # spect_carrier, phase_carrier = spect_carrier.unsqueeze(0), phase_carrier.unsqueeze(0)
    msg_s = [spect_loader(path, trim_start, return_phase=True, num_samples=num_samples) for path in msg_wav_paths]
    for i in msg_s:
      print('Messages: ',i.shape)

    # spects_msg, phases_msg = [D[0].unsqueeze(0) for D in magphase_msg], [D[1].unsqueeze(0) for D in magphase_msg]

    carrier = carrier.to('cuda')
    msg = [spect_msg.to('cuda') for spect_msg in msg_s]

    spect_carrier_reconst, spects_msg_reconst = solver.forward(carrier, msg)

    print('Carrier_recon: ',spect_carrier_reconst.shape)
    for i in spects_msg_reconst:
      print('Messages_recon: ',i.shape)


    spect_carrier_reconst = spect_carrier_reconst.cpu()
    spects_msg_reconst = [spect_msg_reconst.cpu() for spect_msg_reconst in spects_msg_reconst]

    stft = STFT(N_FFT, HOP_LENGTH)
    out_carrier = stft.inverse(spect_carrier_reconst).detach().numpy()
    print('New_time_audi_carr: ',out_carrier.shape)
    print(out_carrier.squeeze(0).shape)
    orig_out_carrier = stft.inverse(carrier.cpu()).detach().numpy()
    print(orig_out_carrier.shape)

    outs_msg = [stft.inverse(spect_msg_reconst).detach().numpy() for spect_msg_reconst in spects_msg_reconst]
    for i in outs_msg:
      print('New_time_audi_msg: ',i.shape)

    orig_outs_msg = [stft.inverse(spect_msg.cpu()).detach().numpy() for spect_msg in msg]
    # outs_msg_gl = [griffin_lim(m.cpu(), n_iter=50)[0, 0].detach().numpy() for m in spects_msg_reconst]

    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_embedded.wav'), out_carrier.squeeze(0), sr,'PCM_24')
    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_orig.wav'), orig_out_carrier.squeeze(0), sr,'PCM_24')
    for i in range(len(outs_msg)):
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_recovered_orig_.wav'), outs_msg[i].squeeze(0), sr,'PCM_24')
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_orig.wav'), orig_outs_msg[i].squeeze(0), sr,'PCM_24')
        # sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_recovered_gl_phase.wav'), outs_msg_gl[i], sr)
