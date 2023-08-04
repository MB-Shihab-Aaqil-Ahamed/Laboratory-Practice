Fs = 6000; %The sampling frequency used to sample the audio
qbits = 8; %The number of bits used to encode a single sample of the audio
[audio_samples,Fs] = audioread('Recording.wav'); %Replace filename with the location of your file.

siz = size(audio_samples); %Quantizing the audio samples (Each sample is quantized with 8 bits)
size_audio_samples = siz(1);
number_of_bits = qbits*size_audio_samples
bit_stream = zeros([1,number_of_bits]); %array which is used to store the binary representation
%of the recorded audio
for i = 1:size_audio_samples
    num = audio_samples(i,1);
    if(num<0)
        bit_stream(1,(i-1)*qbits+1) = 1;
    else
        bit_stream(1,(i-1)*qbits+1) = 0;
    end
    num = abs(num);
    for j = 2:qbits
        num = 2*num;
        if(num>=1)
            bit_stream(1,(i-1)*qbits+j) = 1;
            num = num - 1;
        else
            bit_stream(1,(i-1)*qbits+j) = 0;
        end
    end
end
samples_per_bit = 100;
Amplitude = -5; %Amplitude in decibels
sampling_rate = Fs*qbits*samples_per_bit; %The sampling rate used to represent the
%anlog singal in MATLAB
fc = sampling_rate/100; % frequency of the sinosoid
Wc = 2*pi*fc;
delt = 1/sampling_rate; %Sample time period
phase0 = pi;
phase1 = 0;
number_of_samples = number_of_bits*samples_per_bit; %Total number of samples of the transmitted signal
X = zeros([1,3*number_of_samples]); %X(t) − This array is used to store the transmitted analog signal
power = 10^(Amplitude/20); %Signal power in watts.
t_bit = delt:delt:delt*samples_per_bit;
bit1 = sqrt(power)*sin(Wc*t_bit+phase1); %Representation of bit 1
bit0 = sqrt(power)*sin(Wc*t_bit+phase0); %Representation of bit 0
for i = 1:number_of_bits
    if(bit_stream(i)==1)
        X(3*samples_per_bit*(i-1)+1:3*samples_per_bit*(i-1)+samples_per_bit) = bit1;
        X(3*samples_per_bit*(i-1)+samples_per_bit+1:3*samples_per_bit*(i-1)+2*samples_per_bit) = bit1;
        X(3*samples_per_bit*(i-1)+2*samples_per_bit+1:3*samples_per_bit*i) = bit1;
    else
        X(3*samples_per_bit*(i-1)+1:3*samples_per_bit*(i-1)+samples_per_bit) = bit0;
        X(3*samples_per_bit*(i-1)+samples_per_bit+1:3*samples_per_bit*(i-1)+2*samples_per_bit) = bit0;
        X(3*samples_per_bit*(i-1)+2*samples_per_bit+1:3*samples_per_bit*i) = bit0;
    end
end
att = 0.8;%attenuation factor
mu = 0; %Paramenters of the noise signal (mu, and sigma). We
% model noise as a Gaussian random variable
sigma = 1;
N = normrnd(mu,sigma,1,3*number_of_samples); %N(t) − noise signal
X_hat = att*X + N; %signal after noise
fc_butter = fc*25;
[fil_b,fil_a] = butter(10,fc_butter/(sampling_rate/2));
Y = filter(fil_b,fil_a,X_hat);%recived signal after bandwidth limitation (filtering)
decoded_bit_stream = zeros([1,3*number_of_bits]); %The array to store the decoded bit stream
H = [cos(Wc*delt*[1:samples_per_bit]'),sin(Wc*delt*[1:samples_per_bit]')]; %Phase estimation matrox
for i = 1:3*number_of_bits %This loop deals with the phase estimation
    f = inv(H'*H)*H'*Y((i-1)*samples_per_bit+1:i*samples_per_bit)';
    decoded_phase = 0;
    if(f(2) == 0)
        if(f(1) >= 0)
            decoded_phase = pi/2;
        else
            decoded_phase = -pi/2;
        end
    else
        decoded_phase = atan(f(1)/f(2));
        if(f(1)<=0 & f(2)<0)
            decoded_phase = decoded_phase-pi;
        elseif(f(1)>0 & f(2)<0)
            decoded_phase = decoded_phase+pi;
        end
    end
    p0 = min(abs(decoded_phase - phase0),2*pi-abs(decoded_phase - phase0));
    p1 = min(abs(decoded_phase - phase1),2*pi-abs(decoded_phase - phase1));
    if(p0<p1)
        decoded_bit_stream(i) = 0;
    else
        decoded_bit_stream(i) = 1;
    end
end
decoded_bit_stream_error_corrected = zeros([1,number_of_bits]); %The array to store the decoded bit
%stream after error−correction
for i = 1:number_of_bits %This loop does the task of correcting erros by looking at the most frequent
    %bit from the three bits
    for j = 1:3
        decoded_bit_stream_error_corrected(i) = decoded_bit_stream_error_corrected(i)+decoded_bit_stream(3*(i-1)+j);
    end
    decoded_bit_stream_error_corrected(i)= round(decoded_bit_stream_error_corrected(i)/3);
end
decoded_audio_samples = zeros([size_audio_samples,1]);
for i=1:size_audio_samples
    st = 0.5;
    for j=2:qbits
        decoded_audio_samples(i) = decoded_audio_samples(i)+st*decoded_bit_stream_error_corrected((i-1)*qbits+j);
        st = st/2;
    end
    if(decoded_bit_stream_error_corrected((i-1)*qbits+1) == 1)
        decoded_audio_samples(i) = -decoded_audio_samples(i);
    end
end

pause(10);
sound(5*decoded_audio_samples, Fs);

