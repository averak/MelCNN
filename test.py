## ========================
import scipy.io.wavfile as wf
spec = stft(w1, CONFIG['wave']['fs'], False).T
mask = np.array(y)
melcnn = MelCNN(spec.shape[1])
wav = melcnn.vocoder(spec.T, mask.T, True)
print(wav)
ori = np.array(w1, dtype='int16')
wf.write('分離後.wav', CONFIG['wave']['fs'], wav*100)
wf.write('分離前.wav', CONFIG['wave']['fs'], ori*100)
exit(0)
## ========================
