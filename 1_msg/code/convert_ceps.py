#this is problematic

from loguru import logger
from os.path import join, basename
from hparams import *
from stft.stft import STFT
from dataloader import spect_loader
import soundfile as sf
from griffin_lim import griffin_lim

def convert(solver, carrier_wav_path, msg_wav_paths, trg_dir, epoch, trim_start, num_samples):
    if solver.mode != 'test':
        logger.warning("generating audio not in test mode!")

    _, sr = sf.read(carrier_wav_path)
    carrier_basename = basename(carrier_wav_path).split(".")[0]
    msg_basenames = [basename(msg_wav_path).split(".")[0] for msg_wav_path in msg_wav_paths]

    spect_carrier = spect_loader(carrier_wav_path, trim_start, return_phase=True, num_samples=num_samples)
    magphase_msg = [spect_loader(path, trim_start, return_phase=True, num_samples=num_samples) for path in msg_wav_paths]
    spects_msg= [D for D in magphase_msg]

    spect_carrier = spect_carrier.to('cuda')
    spects_msg = [spect_msg.to('cuda') for spect_msg in spects_msg]
    spects_msg = torch.cat(spects_msg, dim=1)

    spect_carrier_reconst, spects_msg_reconst = solver.forward(spect_carrier, spects_msg)
    spect_carrier_reconst = spect_carrier_reconst.cpu()
    spects_msg_reconst = spect_msg_reconst.cpu()

    stft = STFT(N_FFT, HOP_LENGTH)
    out_carrier = stft.inverse(spect_carrier_reconst).detach().numpy()
    orig_out_carrier = stft.inverse(spect_carrier.cpu()).detach().numpy()

    outs_msg = stft.inverse(spect_msg_reconst).detach().numpy() for spect_msg_reconst in spects_msg_reconst
    orig_outs_msg = [stft.inverse(spect_msg.cpu()).detach().numpy() for spect_msg in spects_msg]
    #outs_msg_gl = [griffin_lim(m.cpu(), n_iter=50)[0, 0].detach().numpy() for m in spects_msg_reconst]

    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_embedded.wav'), out_carrier, sr)
    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_orig.wav'), orig_out_carrier, sr)
    sf.write(join(trg_dir, f'{epoch}_{msg_basenames}_msg_recovered_orig_phase.wav'), outs_msg, sr)