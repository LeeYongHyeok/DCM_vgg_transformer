import torch

def get_audio_video_mask(audio_encoder_output, video_encoder_output, windowing=6, av_ratio=4):
    audio_length = audio_encoder_output.size(0)
    video_length = video_encoder_output.size(0)
    
    video_padding_mask = torch.zeros([video_length,audio_length]).to(audio_encoder_output.get_device())
    
    for x_idx in range(video_length):
        st = x_idx*av_ratio - (windowing/2)*av_ratio
        ed = (x_idx+1)*av_ratio + (windowing/2)*av_ratio
        video_padding_mask[x_idx, int(max(st,0)):int(min(ed,audio_length))]=1
    
    video_padding_mask = video_padding_mask.float().masked_fill(video_padding_mask == 0, float('-inf')).masked_fill(video_padding_mask == 1, float(0.0))
    audio_padding_mask = video_padding_mask.transpose(0,1)

    return audio_padding_mask, video_padding_mask
    
