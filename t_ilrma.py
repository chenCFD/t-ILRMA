
import wave as wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import time


def execute_ip_time_varying_gaussian_tilrma(x,W,a,b,mu,p,n_iterations=20):

    for t in range(n_iterations):
        
        s_hat=np.einsum('kmn,nkt->mkt',W,x)
        s_power=np.square(np.abs(s_hat)) 

        v=np.einsum("bst,ksb->skt",a,b)
        
        sigma=(np.maximum(v,1.e-18))**(1/p)
        alpha=1+(2/mu)*(s_power/(np.maximum(v,1.e-18))**(2/p))
        k=alpha*(sigma**2)
        sp=sigma**p

        a=a*(((np.einsum("ksb,skt->bst",b,s_power/(((mu/(mu+2))*(sigma**2+s_power))*np.maximum(sp,1.e-18)))/np.einsum("ksb,skt->bst",b,1./np.maximum(sp,1.e-18)))**(p/(p+2)))*0.4)

        b=b*(((np.einsum("bst,skt->ksb",a,s_power/(((mu/(mu+2))*(sigma**2+s_power))*np.maximum(sp,1.e-18)))/np.einsum("bst,skt->ksb",a,1./np.maximum(sp,1.e-18)))**(p/(p+2)))**0.4)

        v=np.einsum("bst,ksb->skt",a,b)
        
        sigma=(np.maximum(v,1.e-18))**(1/p)
        alpha=1+(2/mu)*(s_power/(np.maximum(v,1.e-18))**(2/p))
        k=alpha*(sigma**2)

        Q=(2/mu+1)*np.einsum('skt,mkt,nkt->tksmn',1./np.maximum(k,1.e-18),x,np.conjugate(x))
        Q=np.average(Q,axis=0)
        
        for source_index in range(np.shape(x)[0]):
            WQ=np.einsum('kmi,kin->kmn',W,Q[:,source_index,:,:])
            invWQ=np.linalg.pinv(WQ)
            W[:,source_index,:]=np.conjugate(invWQ[:,:,source_index])
            wVw=np.einsum('km,kmn,kn->k',W[:,source_index,:],Q[:,source_index,:,:],np.conjugate(W[:,source_index,:]))
            wVw=np.sqrt(np.abs(wVw))
            W[:,source_index,:]=W[:,source_index,:]/np.maximum(wVw[:,None],1.e-18)

 
    s_hat=np.einsum('kmn,nkt->mkt',W,x)

    return(W,s_hat)
    

def projection_back(s_hat,W):

    A=np.linalg.pinv(W)
    c_hat=np.einsum('kmi,ikt->mikt',A,s_hat)
    return(c_hat)


def write_file_from_time_signal(signal,file_name,sample_rate):

    signal=signal.astype(np.int16)


    wave_out = wave.open(file_name, 'w')

    wave_out.setnchannels(1)

    wave_out.setsampwidth(2)

    wave_out.setframerate(sample_rate)

    wave_out.writeframes(signal)

    wave_out.close()


def calculate_sisdr(desired,out):
    wave_length=np.minimum(np.shape(desired)[0],np.shape(out)[0])

    desired=desired[:wave_length]
    out=out[:wave_length]
    d=(np.dot(desired,out)/np.sum(np.square(desired)))*desired
    noise=out-d
    sisdr=10.*np.log10(np.sum(np.square(d))/np.sum(np.square(noise)))

    return(sisdr)


np.random.seed(0)

clean_wave_files=["./sound1.wav","./sound2.wav"]

n_sources=len(clean_wave_files)

n_samples=0

for clean_wave_file in clean_wave_files:
    wav=wave.open(clean_wave_file)
    if n_samples<wav.getnframes():
        n_samples=wav.getnframes()
    wav.close()

clean_data=np.zeros([n_sources,n_samples])

s=0
for clean_wave_file in clean_wave_files:
    wav=wave.open(clean_wave_file)
    data=wav.readframes(wav.getnframes())
    data=np.frombuffer(data, dtype=np.int16)
    data=data/np.iinfo(np.int16).max
    clean_data[s,:wav.getnframes()]=data
    wav.close()
    s=s+1


n_sim_sources=2

sample_rate=40000

N=1024

Nk=int(N/2+1)

freqs=np.arange(0,Nk,1)*sample_rate/N

SNR=90.

room_dim = np.r_[10.0, 10.0, 10.0]
k=np.r_[0.0, 1.0, 0.0]

mic_array_loc0 = room_dim / 2 + np.random.randn(3) * 0.1 
mic_array_loc=mic_array_loc0+k

mic_directions=np.array(
    [[np.pi/2., theta/180.*np.pi] for theta in np.arange(180,361,180)
    ]    )

distance=0.01
mic_alignments=np.zeros((3, mic_directions.shape[0]), dtype=mic_directions.dtype)
mic_alignments[0, :] = np.cos(mic_directions[:, 1]) * np.sin(mic_directions[:, 0])
mic_alignments[1, :] = np.sin(mic_directions[:, 1]) * np.sin(mic_directions[:, 0])
mic_alignments[2, :] = np.cos(mic_directions[:, 0])
mic_alignments *= distance

n_channels=np.shape(mic_alignments)[1]

R=mic_alignments+mic_array_loc[:,None]

is_use_reverb=False

if is_use_reverb==False:

    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0) 
    room_no_noise_left = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
    room_no_noise_right = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)

else:

    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=3,absorption=0.4)
    room_no_noise_left = pa.ShoeBox(room_dim, fs=sample_rate, max_order=3,absorption=0.4)
    room_no_noise_right = pa.ShoeBox(room_dim, fs=sample_rate, max_order=3,absorption=0.4)

room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
room_no_noise_left.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
room_no_noise_right.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))

doas=np.array(
    [[np.pi/2., np.pi],
     [np.pi/2., 0]
    ]    )

distance=1.

source_locations=np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance
source_locations += mic_array_loc0[:, None]

for s in range(n_sim_sources):
    clean_data[s]/= np.std(clean_data[s])
    room.add_source(source_locations[:, s], signal=clean_data[s])
    if s==0:
        room_no_noise_left.add_source(source_locations[:, s], signal=clean_data[s])
    if s==1:
        room_no_noise_right.add_source(source_locations[:, s], signal=clean_data[s])

room.simulate(snr=SNR)
room_no_noise_left.simulate(snr=90)
room_no_noise_right.simulate(snr=90)

room.plot()
plt.show()

multi_conv_data=room.mic_array.signals
multi_conv_data_left_no_noise=room_no_noise_left.mic_array.signals
multi_conv_data_right_no_noise=room_no_noise_right.mic_array.signals

write_file_from_time_signal(multi_conv_data_left_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./tilrma_left_clean.wav",sample_rate)

write_file_from_time_signal(multi_conv_data_right_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./tilrma_right_clean.wav",sample_rate)

write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./tilrma_in_left.wav",sample_rate)
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./tilrma_in_right.wav",sample_rate)

f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

n_tilrma_iterations=20

mu=100000
p=1

n_basis=2

Lt=np.shape(stft_data)[-1]

Wtilrma_ip=np.zeros(shape=(Nk,n_sources,n_sources),dtype=np.complex)

Wtilrma_ip=Wtilrma_ip+np.eye(n_sources)[None,...]

b=np.ones(shape=(Nk,n_sources,n_basis))
a=np.random.uniform(size=(n_basis*n_sources*Lt))
a=np.reshape(a,(n_basis,n_sources,Lt))

start_time=time.time()


Wtilrma_ip,s_tilrma_ip=execute_ip_time_varying_gaussian_tilrma(stft_data,Wtilrma_ip,a,b,mu,p,n_iterations=n_tilrma_iterations)
y_tilrma_ip=projection_back(s_tilrma_ip,Wtilrma_ip)
tilrma_ip_time=time.time()

t,y_tilrma_ip=sp.istft(y_tilrma_ip[0,...],fs=sample_rate,window="hann",nperseg=N)

sisdr_pre=calculate_sisdr(multi_conv_data_left_no_noise[0,...],multi_conv_data[0,...])+calculate_sisdr(multi_conv_data_right_no_noise[0,...],multi_conv_data[0,...])
sisdr_pre/=2.


sisdr_tilrma_ip_post1=calculate_sisdr(multi_conv_data_left_no_noise[0,...],y_tilrma_ip[0,...])+calculate_sisdr(multi_conv_data_right_no_noise[0,...],y_tilrma_ip[1,...])
sisdr_tilrma_ip_post2=calculate_sisdr(multi_conv_data_left_no_noise[0,...],y_tilrma_ip[1,...])+calculate_sisdr(multi_conv_data_right_no_noise[0,...],y_tilrma_ip[0,...])

sisdr_tilrma_ip_post=np.maximum(sisdr_tilrma_ip_post1,sisdr_tilrma_ip_post2)
sisdr_tilrma_ip_post/=2.


write_file_from_time_signal(y_tilrma_ip[0,...]*np.iinfo(np.int16).max/20.,"./t_ilrma_ip_1.wav",sample_rate)
write_file_from_time_signal(y_tilrma_ip[1,...]*np.iinfo(np.int16).max/20.,"./t_ilrma_ip_2.wav",sample_rate)

print("method:    ", "t-ILRMA")
print("処理時間[sec]: {:.2f}".format(tilrma_ip_time-start_time))
print("Δsisdr [dB]: {:.2f}".format(sisdr_tilrma_ip_post-sisdr_pre))