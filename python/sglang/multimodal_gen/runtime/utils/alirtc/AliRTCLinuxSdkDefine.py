from enum import Enum
from typing import List, Tuple

class ERROR_CODE(Enum):
    ERR_OK = 0,
    # //////////// 加入频道错误码 ////////////
    ERR_JOIN_BAD_APPID                         = 0x02010201 # AppId不存在
    ERR_JOIN_INVALID_APPID                     = 0x02010202 # AppId已失效
    ERR_JOIN_BAD_CHANNEL                       = 0x02010204 # 频道不存在
    ERR_JOIN_INVALID_CHANNEL                   = 0x02010203 # 频道已失效
    ERR_JOIN_BAD_TOKEN                         = 0x02010205 # Token不存在
    ERR_JOIN_TIMEOUT                           = 0x01020204 # 加入频道超时
    ERR_JOIN_BAD_PARAM                         = 0x01030101 # 加入频道参数错误
    ERR_JOIN_FAILED                            = 0x01030202 # 加入频道失败
    ERR_JOIN_CONFIG_INVALID                    = 0x01030302 # 入会参数无效，如空token
    ERR_LEAVE_UNUSUAL                          = 0x01030303 # 异常离会，可能被踢出房间
    ERR_NO_PEOPLE                              = 0x01030304 # 房间无人，可能房间不存在或主播已下播
    # //////////// 媒体错误码 ////////////
    ERR_AUDIO_DATA_ERROR                       = 0x03010201 # 音频数据错误
    ERR_VIDEO_DATA_ERROR                       = 0x03010202 # 视频数据错误
    # ///////// 网络错误码 ////////////
    ERR_NETWORK_CONNECT_FAIL                   = 0x01050201 # 媒体通道建立失败
    ERR_NETWORK_RECONNECT_FAIL                 = 0x01050202 # 媒体通道重连失败
    ERR_NETWORK_DISCONNECT                     = 0x01050203 # 连接断开
    ERR_NETWORK_TIMEOUT                        = 0x0102020C # 连接超时
    # ///////// 推流相关错误码 ////////////
    ERR_PUBLISH_INVALID                        = 0x01030305 # 推流无效
    ERR_Publish_NOT_JOIN_CHANNEL               = 0x01010406 # 未进入频道推流失败
    ERR_PUBLISH_AUDIO_STREAM_FAILED            = 0x01010450 # 推送音频流失败
    ERR_PUBLISH_VIDEO_STREAM_FAILED            = 0x01010451 # 推送视频流失败
    ERR_PUBLISH_DUAL_STREAM_FAILED             = 0x01010452 # 推送小流失败
    ERR_PUBLISH_SCREEN_SHARE_FAILED            = 0x01010453 # 推送屏幕共享失败
    ERR_PUBLISH_SCREEN_SHARE_CONFIG            = 0x01010454 # 屏幕共享配置错误
    # ///////// 订阅相关错误码 ////////////    
    ERR_SUBSCRIBE_INVALID                      = 0x01030404 # 订阅无效    
    ERR_SUBSCRIBE_NOT_JOIN_CHANNEL             = 0x01010550 # 未进入频道订阅错误
    ERR_SUBSCRIBE_AUDIO_STREAM_FAILED          = 0x01010551 # 订阅音频流失败
    ERR_SUBSCRIBE_VIDEO_STREAM_FAILED          = 0x01010552 # 订阅视频流失败
    ERR_SUBSCRIBE_DUAL_STREAM_FAILED           = 0x01010553 # 订阅小流失败
    ERR_SUBSCRIBE_SCREEN_SHARE_FAILED          = 0x01010554 # 订阅屏幕共享失败
    ERR_SUBSCRIBE_DUAL_AUDIO_STREAM_FAILED     = 0x01010555 # 订阅双声道失败
    # ///////// 其他错误码 ////////////
    ERR_SDK_INVALID_STATE                      = 0x01060101 # SDK内部状态错误
    ERR_SESSION_REMOVED                        = 0x01060102 # session已经被移除
    ERR_INNER                                  = 0x01060103 # SDK内部错误
    ERR_VIDEO_TRANSFER                         = 0x01060104 # 视频数据传输错误
    ERR_AUDIO_TRANSFER                         = 0x01060105 # 音频数据传输错误
    ERR_AUDIO_BUFFER_FULL                      = 0x01060106 # 音频外部输入时，频率过快
    ERR_FATAL                                  = -1         # SDK严重错误


class WARNNING_CODE(Enum):
    WARN_DEFAULT = -1


class ChannelProfile(Enum):
    ChannelProfileCommunication = 0                       # 通信模式
    ChannelProfileInteractiveLive = 1                     # 直播模式
    ChannelProfileInteractiveWithLowLatencyLive = 2


class PublishMode(Enum):
    PublishAutomatically = 0
    PublishManually = 1



# 日志级别
class LogLevel(Enum):
    LogLevelInfo  = 3  # 只输出>=AliEngineLogLevelInfo 级别的日志
    LogLevelWarn  = 4  # 只输出>=AliEngineLogLevelWarn 级别的日志
    LogLevelError = 5  # 只输出>=AliEngineLogLevelError 级别的日志
    LogLevelFatal = 6  # 只输出>=AliEngineLogLevelFatal 级别的日志
    LogLevelNone  = 7


class AudioQualityMode(Enum):
    LowQualityMode = 0x0000             # 音频低音质模式，默认8000Hz采样率，单声道，最大编码码率12kbps
    BasicQualityMode = 0x0001           #（默认）标准音质模式，默认16000Hz采样率，单声道，最大编码码率24kbps
    HighQualityMode = 0x0010            # 高音质模式，默认48000Hz采样率，单声道，最大编码码率64kbps
    StereoHighQualityMode = 0x0011      # 立体声高音质模式，默认48000Hz采样率，双声道，最大编码码率80kbps
    SuperHighQualityMode = 0x0012       # 超高音质模式，默认48000Hz采样率，单声道，最大编码码率96kbps
    StereoSuperHighQualityMode = 0x0013 # 立体声超高音质模式，默认48000Hz采样率，双声道，最大编码码率128kbps
    QualityMaxMode = 0x0014


class AudioSceneMode(Enum):
    DefaultMode                      = 0x0000   # hw 3a
    EducationMode                    = 0x0100   # hw 3a
    MediaMode                        = 0x0200   # basic sw 3a
    MusicMode                        = 0x0300   # music modify with sw 3a
    ChatroomMode                     = 0x0400   # hw 3a
    KtvMode                          = 0x0500   # music modify with sw 3a + Aaudio
    SceneMaxMode                     = 0x0501


class VideoTrack(Enum):
    VideoTrackNo     = 0  # no video track
    VideoTrackCamera = 1  # video from camera, file, etc.
    VideoTrackScreen = 2  # video from screen sharing
    VideoTrackBoth   = 3  # both VideoTrackCamera and VideoTrackScreen
    VideoTrackEnd    = 0xffffffff


class AudioTrack(Enum):
    AudioTrackNo  = 0  # no audio track
    AudioTrackMic = 1  # audio from mic, file, etc.
    AudioTrackEnd = 0xffffffff


class VideoProfile(Enum):
    VideoProfile_Default = 0  # let sdk decide
    VideoProfile_180_240P_15 = 1
    VideoProfile_180_320P_15 = 2
    VideoProfile_180_320P_30 = 3
    VideoProfile_240_320P_15 = 4
    VideoProfile_360_480P_15 = 5
    VideoProfile_360_480P_30 = 6
    VideoProfile_360_640P_15 = 7
    VideoProfile_360_640P_30 = 8
    VideoProfile_480_640P_15 = 9
    VideoProfile_480_640P_30 = 10
    VideoProfile_720_960P_15 = 11
    VideoProfile_720_960P_30 = 12
    VideoProfile_720_1280P_15 = 13
    VideoProfile_720_1280P_20 = 14
    VideoProfile_720_1280P_30 = 15
    VideoProfile_1080_1920P_15 = 16
    VideoProfile_1080_1920P_30 = 17
    VideoProfile_480_640P_15_1500Kb = 18
    VideoProfile_900_1600P_20 = 19
    VideoProfile_360_640P_15_800Kb = 20
    VideoProfile_480_840P_15_500Kb = 21
    VideoProfile_480_840P_15_800Kb = 22
    VideoProfile_540_960P_15_800Kb = 23
    VideoProfile_540_960P_15_1200Kb = 24
    VideoProfile_540_960P_20 = 25
    VideoProfile_1080_1920P_20 = 26
    VideoProfile_240_320P_15_300kb = 27
    VideoProfile_240_320P_15_500kb = 28
    VideoProfile_90_160P_15 = 29
    AliRTCSDK_Video_Profile_Adaptive_Resolution = 30
    AliRTCSDK_Video_Profile_Max = 31


# 视频分辨率
class AliEngineVideoDimensions(object):
    width: int
    height: int
    def __init__(self, width:int, height:int) -> None:
        self.width = width
        self.height = height


# 视频帧率
class AliEngineFrameRate(Enum):
    AliEngineFrameRateFps5 = 5     # 5: 5 fps
    AliEngineFrameRateFps10 = 10   # 10: 10 fps
    AliEngineFrameRateFps15 = 15   # 15: 15 fps
    AliEngineFrameRateFps20 = 20   # 20: 20 fps
    AliEngineFrameRateFps25 = 25   # 25: 25 fps
    AliEngineFrameRateFps30 = 30   # 30: 30 fps
    AliEngineFrameRateFps60 = 60   # 60: 60 fps


# 视频输出方向
class AliEngineVideoEncoderOrientationMode(Enum):
    '''
        0: 自适应，推流方向和采集方向一致
    '''
    AliEngineVideoEncoderOrientationModeAdaptive = 0
    '''
        1: 风景模式，横屏视频
        该模式下SDK推竖屏流，始终以设置的分辨率宽和高中较小的值作为输出视频的宽，较大值作为输出视频的高
    '''
    AliEngineVideoEncoderOrientationModeFixedLandscape = 1
    '''
        2: 肖像模式，竖屏视频
        该模式下SDK推横屏流，始终以设置的分辨率宽和高中较大的值作为输出视频的宽，较小值作为输出视频的高
    '''
    AliEngineVideoEncoderOrientationModeFixedPortrait = 2

# Qos反馈外置编码器参数结构体
class AliEngineVideoExternalEncoderParameter(object):
    width: int
    height: int
    frame_rate: int
    bitrate_bps: int
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

class AliEngineVideoEncodedFrameType(Enum):
    AliEngineVideoEncodedFrameNULL     = 0  # 默认 无
    AliEngineVideoEncodedFrameIDR = 1  # IDR帧
    AliEngineVideoEncodedFrameLTR = 2  # LTR帧
    AliEngineVideoEncodedFrameB   = 3  # B帧

# 视频镜像模式
class AliEngineVideoMirrorMode(Enum):
    '''
        0:  关闭镜像
    '''
    AliEngineVideoMirrorModeDisabled = 0 # disable mirror
    '''
        1:  开启镜像
    '''
    AliEngineVideoMirrorModeEnable = 1   # enabled mirror


# 视频旋转角度
class AliEngineRotationMode(Enum):
    # 沿用之前的旋转角度
    AliEngineRotationModeNoChange = -1
    # 0
    AliEngineRotationMode_0 = 0
    # 90
    AliEngineRotationMode_90 = 90
    # 180
    AliEngineRotationMode_180 = 180
    # 270
    AliEngineRotationMode_270 = 270


# 相机流视频编码属性设置
class AliEngineVideoEncoderConfiguration(object):
    '''
        视频分辨率，默认值640x480，最大值1920x1080
    '''
    dimensions: AliEngineVideoDimensions
    
    '''
        视频帧率，默认值15, 最大值30
    '''
    frameRate: AliEngineFrameRate

    '''
        Gop大小，单位毫秒
    '''
    keyFrameInterval: int
    
    '''
     视频编码码率(Kbps)
    - 默认值 512
    - 设置为0，表示由SDK内部根据视频分辨率和码率计算合适的编码码率
    - 码率设置根据分辨率和帧率有对应的合理范围，该值设置在合理范围内有效，否则SDK会自动调节码率到有效值
        
    @note
    以下码表列举常见的分辨率、帧率对应的编码码率设置的区间

        | 分辨率          | 帧率(fps)       | 最小码率 (Kbps) | 推荐码率(Kbps)  |最大码率(Kbps)
        |----------------|----------------|----------------|----------------|----------------
        | 120 * 120      | 5              | 10             | 25             | 75
        | 120 * 120      | 10             | 17             | 50             | 150
        | 120 * 120      | 15             | 25             | 70             | 210
        | 120 * 120      | 20             | 34             | 90             | 270
        | 120 * 120      | 30             | 50             | 115            | 345
        | 160 * 120      | 5              | 10             | 30             | 90
        | 160 * 120      | 10             | 20             | 55             | 165
        | 160 * 120      | 15             | 30             | 80             | 240
        | 160 * 120      | 20             | 40             | 100            | 300
        | 160 * 120      | 30             | 60             | 130            | 390
        | 180 * 180      | 5              | 10             | 50             | 150
        | 180 * 180      | 10             | 17             | 70             | 210
        | 180 * 180      | 15             | 26             | 100            | 300
        | 180 * 180      | 20             | 34             | 130            | 390
        | 180 * 180      | 30             | 51             | 180            | 540
        | 240 * 180      | 5              | 15             | 60             | 180
        | 240 * 180      | 10             | 30             | 90             | 270
        | 240 * 180      | 15             | 45             | 130            | 390
        | 240 * 180      | 20             | 60             | 165            | 495
        | 240 * 180      | 30             | 90             | 230            | 690
        | 320 * 180      | 5              | 15             | 65             | 195
        | 320 * 180      | 10             | 30             | 110            | 330
        | 320 * 180      | 15             | 45             | 170            | 510
        | 320 * 180      | 20             | 60             | 220            | 660
        | 320 * 180      | 30             | 90             | 300            | 900
        | 240 * 240      | 5              | 15             | 70             | 140
        | 240 * 240      | 10             | 30             | 100            | 200
        | 240 * 240      | 15             | 45             | 150            | 300
        | 240 * 240      | 20             | 60             | 200            | 400
        | 240 * 240      | 30             | 90             | 256            | 512
        | 320 * 240      | 5              | 20             | 100            | 200
        | 320 * 240      | 10             | 40             | 170            | 340
        | 320 * 240      | 15             | 60             | 256            | 512
        | 320 * 240      | 20             | 80             | 320            | 640
        | 320 * 240      | 30             | 120            | 400            | 800
        | 424 * 240      | 5              | 26             | 100            | 200
        | 424 * 240      | 10             | 53             | 170            | 340
        | 424 * 240      | 15             | 79             | 260            | 520
        | 424 * 240      | 20             | 105            | 340            | 680
        | 424 * 240      | 30             | 158            | 430            | 860
        | 360 * 360      | 5              | 30             | 120            | 240
        | 360 * 360      | 10             | 60             | 180            | 360
        | 360 * 360      | 15             | 90             | 260            | 520
        | 360 * 360      | 20             | 120            | 330            | 660
        | 360 * 360      | 30             | 180            | 400            | 800
        | 480 * 360      | 5              | 40             | 150            | 300
        | 480 * 360      | 10             | 80             | 240            | 480
        | 480 * 360      | 15             | 120            | 350            | 700
        | 480 * 360      | 20             | 160            | 430            | 860
        | 480 * 360      | 30             | 240            | 512            | 1024
        | 640 * 360      | 5              | 83             | 200            | 400
        | 640 * 360      | 10             | 165            | 340            | 680
        | 640 * 360      | 15             | 248            | 512            | 1024
        | 640 * 360      | 20             | 330            | 600            | 1200
        | 640 * 360      | 30             | 495            | 700            | 1400
        | 480 * 480      | 5              | 83             | 170            | 340
        | 480 * 480      | 10             | 165            | 260            | 520
        | 480 * 480      | 15             | 248            | 400            | 800
        | 480 * 480      | 20             | 330            | 470            | 940
        | 480 * 480      | 30             | 495            | 600            | 1200
        | 640 * 480      | 5              | 110            | 200            | 400
        | 640 * 480      | 10             | 220            | 350            | 700
        | 640 * 480      | 15             | 330            | 512            | 1024
        | 640 * 480      | 20             | 440            | 600            | 1200
        | 640 * 480      | 30             | 660            | 700            | 1400
        | 840 * 480      | 5              | 180            | 256            | 512
        | 840 * 480      | 10             | 360            | 512            | 1024
        | 840 * 480      | 15             | 540            | 610            | 1220
        | 840 * 480      | 20             | 720            | 800            | 1600
        | 840 * 480      | 30             | 1080           | 930            | 1860
        | 960 * 720      | 5              | 250            | 250            | 600
        | 960 * 720      | 10             | 500            | 500            | 750
        | 960 * 720      | 15             | 750            | 750            | 1125
        | 960 * 720      | 20             | 1000           | 1000           | 1500
        | 960 * 720      | 30             | 1500           | 1500           | 2250
        | 1280 * 720     | 5              | 400            | 400            | 600
        | 1280 * 720     | 10             | 800            | 800            | 1200
        | 1280 * 720     | 15             | 1200           | 1200           | 1800
        | 1280 * 720     | 20             | 1600           | 1600           | 2400
        | 1280 * 720     | 30             | 2400           | 2400           | 3600
        | 1920 * 1080    | 5              | 500            | 500            | 750
        | 1920 * 1080    | 10             | 1000           | 1000           | 1500
        | 1920 * 1080    | 15             | 1500           | 1500           | 2250
        | 1920 * 1080    | 20             | 2000           | 2000           | 3000
        | 1920 * 1080    | 30             | 3000           | 3000           | 4500
        | 2560 * 1440    | 5              | 800            | 800            | 1200
        | 2560 * 1440    | 10             | 1600           | 1600           | 2400
        | 2560 * 1440    | 15             | 2400           | 2400           | 3600
        | 2560 * 1440    | 20             | 3200           | 3200           | 4800
        | 2560 * 1440    | 30             | 4800           | 4800           | 7200
        | 3840 * 2160    | 5              | 1000           | 1000           | 1500
        | 3840 * 2160    | 10             | 2000           | 2000           | 3000
        | 3840 * 2160    | 15             | 3000           | 3000           | 4500
        | 3840 * 2160    | 20             | 4000           | 4000           | 6000
        | 3840 * 2160    | 30             | 6000           | 6000           | 9000
    '''
    bitrate: int
    minBitrate: int

    '''
        视频输出方向，默认AliEngineVideoEncoderOrientationModeAdaptive，详见 {@link AliEngineVideoEncoderOrientationMode}
    '''
    orientationMode: AliEngineVideoEncoderOrientationMode
    
    '''
        推流镜像，默认AliEngineVideoMirrorModeDisabled，详见 {@link AliEngineVideoMirrorMode}
    '''
    mirrorMode: AliEngineVideoMirrorMode
    
    '''
        推流旋转，默认AliEngineRotationMode_0，详见 {@link AliEngineRotationMode}
    '''
    rotationMode: AliEngineRotationMode
    
    def __init__(self, width:int, height:int, f:AliEngineFrameRate, b:int, \
                 ori:AliEngineVideoEncoderOrientationMode, mr:AliEngineVideoMirrorMode, \
                 rotation:AliEngineRotationMode) -> None:
        self.dimensions = AliEngineVideoDimensions(width, height)
        self.frameRate = f
        self.keyFrameInterval = 0
        self.bitrate = b
        self.minBitrate = 0
        self.orientationMode = ori
        self.mirrorMode = mr
        self.rotationMode = rotation


# 屏幕流编码属性设置
class AliEngineScreenShareEncoderConfiguration(object):
    '''
        视频分辨率，默认值0x0，最大值3840x2160
        @note 默认值表示推流分辨率等于屏幕采集的分辨率
    '''
    dimensions: AliEngineVideoDimensions
    
    
    '''
        视频帧率，默认值5, 最大值30
    '''
    frameRate: AliEngineFrameRate
    
    '''
    视频编码码率(Kbps)
    - 默认值 512
    - 设置为0，表示由SDK内部根据视频分辨率和码率计算合适的编码码率
    - 码率设置根据分辨率和帧率有对应的合理范围，该值设置在合理范围内有效，否则SDK会自动调节码率到有效值
        
    @note
    以下码表列举常见的分辨率、帧率对应的编码码率设置的区间

        | 分辨率          | 帧率(fps)       | 最小码率 (Kbps) | 推荐码率(Kbps)  |最大码率(Kbps)
        |----------------|----------------|----------------|----------------|----------------
        | 120 * 120      | 5              | 10             | 25             | 75
        | 120 * 120      | 10             | 17             | 50             | 150
        | 120 * 120      | 15             | 25             | 70             | 210
        | 120 * 120      | 20             | 34             | 90             | 270
        | 120 * 120      | 30             | 50             | 115            | 345
        | 160 * 120      | 5              | 10             | 30             | 90
        | 160 * 120      | 10             | 20             | 55             | 165
        | 160 * 120      | 15             | 30             | 80             | 240
        | 160 * 120      | 20             | 40             | 100            | 300
        | 160 * 120      | 30             | 60             | 130            | 390
        | 180 * 180      | 5              | 10             | 50             | 150
        | 180 * 180      | 10             | 17             | 70             | 210
        | 180 * 180      | 15             | 26             | 100            | 300
        | 180 * 180      | 20             | 34             | 130            | 390
        | 180 * 180      | 30             | 51             | 180            | 540
        | 240 * 180      | 5              | 15             | 60             | 180
        | 240 * 180      | 10             | 30             | 90             | 270
        | 240 * 180      | 15             | 45             | 130            | 390
        | 240 * 180      | 20             | 60             | 165            | 495
        | 240 * 180      | 30             | 90             | 230            | 690
        | 320 * 180      | 5              | 15             | 65             | 195
        | 320 * 180      | 10             | 30             | 110            | 330
        | 320 * 180      | 15             | 45             | 170            | 510
        | 320 * 180      | 20             | 60             | 220            | 660
        | 320 * 180      | 30             | 90             | 300            | 900
        | 240 * 240      | 5              | 15             | 70             | 140
        | 240 * 240      | 10             | 30             | 100            | 200
        | 240 * 240      | 15             | 45             | 150            | 300
        | 240 * 240      | 20             | 60             | 200            | 400
        | 240 * 240      | 30             | 90             | 256            | 512
        | 320 * 240      | 5              | 20             | 100            | 200
        | 320 * 240      | 10             | 40             | 170            | 340
        | 320 * 240      | 15             | 60             | 256            | 512
        | 320 * 240      | 20             | 80             | 320            | 640
        | 320 * 240      | 30             | 120            | 400            | 800
        | 424 * 240      | 5              | 26             | 100            | 200
        | 424 * 240      | 10             | 53             | 170            | 340
        | 424 * 240      | 15             | 79             | 260            | 520
        | 424 * 240      | 20             | 105            | 340            | 680
        | 424 * 240      | 30             | 158            | 430            | 860
        | 360 * 360      | 5              | 30             | 120            | 240
        | 360 * 360      | 10             | 60             | 180            | 360
        | 360 * 360      | 15             | 90             | 260            | 520
        | 360 * 360      | 20             | 120            | 330            | 660
        | 360 * 360      | 30             | 180            | 400            | 800
        | 480 * 360      | 5              | 40             | 150            | 300
        | 480 * 360      | 10             | 80             | 240            | 480
        | 480 * 360      | 15             | 120            | 350            | 700
        | 480 * 360      | 20             | 160            | 430            | 860
        | 480 * 360      | 30             | 240            | 512            | 1024
        | 640 * 360      | 5              | 83             | 200            | 400
        | 640 * 360      | 10             | 165            | 340            | 680
        | 640 * 360      | 15             | 248            | 512            | 1024
        | 640 * 360      | 20             | 330            | 600            | 1200
        | 640 * 360      | 30             | 495            | 700            | 1400
        | 480 * 480      | 5              | 83             | 170            | 340
        | 480 * 480      | 10             | 165            | 260            | 520
        | 480 * 480      | 15             | 248            | 400            | 800
        | 480 * 480      | 20             | 330            | 470            | 940
        | 480 * 480      | 30             | 495            | 600            | 1200
        | 640 * 480      | 5              | 110            | 200            | 400
        | 640 * 480      | 10             | 220            | 350            | 700
        | 640 * 480      | 15             | 330            | 512            | 1024
        | 640 * 480      | 20             | 440            | 600            | 1200
        | 640 * 480      | 30             | 660            | 700            | 1400
        | 840 * 480      | 5              | 180            | 256            | 512
        | 840 * 480      | 10             | 360            | 512            | 1024
        | 840 * 480      | 15             | 540            | 610            | 1220
        | 840 * 480      | 20             | 720            | 800            | 1600
        | 840 * 480      | 30             | 1080           | 930            | 1860
        | 960 * 720      | 5              | 250            | 250            | 600
        | 960 * 720      | 10             | 500            | 500            | 750
        | 960 * 720      | 15             | 750            | 750            | 1125
        | 960 * 720      | 20             | 1000           | 1000           | 1500
        | 960 * 720      | 30             | 1500           | 1500           | 2250
        | 1280 * 720     | 5              | 400            | 400            | 600
        | 1280 * 720     | 10             | 800            | 800            | 1200
        | 1280 * 720     | 15             | 1200           | 1200           | 1800
        | 1280 * 720     | 20             | 1600           | 1600           | 2400
        | 1280 * 720     | 30             | 2400           | 2400           | 3600
        | 1920 * 1080    | 5              | 500            | 500            | 750
        | 1920 * 1080    | 10             | 1000           | 1000           | 1500
        | 1920 * 1080    | 15             | 1500           | 1500           | 2250
        | 1920 * 1080    | 20             | 2000           | 2000           | 3000
        | 1920 * 1080    | 30             | 3000           | 3000           | 4500
        | 2560 * 1440    | 5              | 800            | 800            | 1200
        | 2560 * 1440    | 10             | 1600           | 1600           | 2400
        | 2560 * 1440    | 15             | 2400           | 2400           | 3600
        | 2560 * 1440    | 20             | 3200           | 3200           | 4800
        | 2560 * 1440    | 30             | 4800           | 4800           | 7200
        | 3840 * 2160    | 5              | 1000           | 1000           | 1500
        | 3840 * 2160    | 10             | 2000           | 2000           | 3000
        | 3840 * 2160    | 15             | 3000           | 3000           | 4500
        | 3840 * 2160    | 20             | 4000           | 4000           | 6000
        | 3840 * 2160    | 30             | 6000           | 6000           | 9000
    '''
    bitrate: int
    
    '''
        推流旋转，默认AliEngineRotationMode_0，详见 {@link AliEngineRotationMode}
    '''
    rotationMode: AliEngineRotationMode
    
    def __init__(self, width, height, frameRate, bitRate, rotation) -> None:
        self.dimensions = AliEngineVideoDimensions(width, height)
        self.frameRate = frameRate
        self.bitrate = bitRate
        self.rotationMode = rotation


class VideoSource(Enum):
    VideoSourceCamera = 0
    VideoSourceScreenShare = 1
    VideosourceTypeMax = 2


class RenderMode(Enum):
    # 自动模式
    RenderModeAuto    = 0    
    # 拉伸平铺模式 ，如果外部输入的视频宽高比和推流设置的宽高比不一致时，将输入视频拉伸到推流设置的比例，画面会变形
    RenderModeStretch = 1    # 延伸
    # 填充黑边模式，如果外部输入的视频宽高比和推流设置的宽高比不一致时，将输入视频上下或者左右填充黑边 
    RenderModeFill    = 2    # 填充
    # 裁剪模式，如果外部输入的视频宽高比和推流设置的宽高比不一致时，将输入视频宽或者高进行裁剪，画面内容会丢失
    RenderModeCrop    = 3    # 裁剪


# 推流音画同步模式
class PublishAvsyncMode(Enum):
    PublishAvsyncNoDelay = 0 # sdk内部不做缓存，PushExternalVideoFrame/PushExternalAudioFrame 后立刻发送到网络，调用方保证音画同步
    PublishAvsyncWithPts = 1 # sdk内部做缓存，根据送入的pts 进行音画对齐和等待
    '''
    PublishAvsyncWithPts下的若干case：
    * 1. 只推音频，但是音频是文件形式输入，比如2s的音频数据，可以快速输入到sdk中，sdk会根据pts按需等待，在2s内陆续输入到网络 
    * 2. 同时输入音视频，但是音视频的获取时间并有一定间隔，sdk内部会根据音频和视频的pts 进行对齐，陆续输入到网络。
    * 在join-leave的生命周期内，多次pub、unpub，送入的不同片段的pts需要在相同时间轴上。
    * 3. join之后，第一次pub，pts 都从0开始，10s后unpub，然后再10s后再进行pub，第二次pub的pts需要从20s开始，不可以再从0开始，否则会导致卡顿。
    '''


class VideoDataFormat(Enum):
    VideoDataFormatUnknow = -1
    VideoDataFormatBGRA = 0
    VideoDataFormatI420 = 1
    VideoDataFormatNV21 = 2
    VideoDataFormatNV12 = 3
    VideoDataFormatH264 = 4
    VideoDataFormatH265 = 5
    VideoDataFormatRGB24 = 6
    VideoDataFormatBGR24 = 7


class VideoBufferType(Enum):
    VideoBufferTypeRawData = 0
    VideoBufferTypeCVPixelBuffer = 1
    VideoBufferTypeTexture = 2


class VideoDataSample(object):
    format:VideoDataFormat
    bufferType:VideoBufferType
    data:bytes
    dataLen:int
    strideY:int
    strideU:int
    strideV:int
    height:int
    width:int
    rotation:int
    timeStamp:int

class AudioFrameData(object):
    data:bytes
    dataLen:int
    timeStamp:int = 0
    frameID:str = ""
    sentenceID:int = -1
    sequenceID:int = -1
    isEndFrame:int = 0

# The Following Define Only Design For Subscribe
class SubscribeMode(Enum):
    SubscribeAutomatically = 0
    SubscribeManually = 1
    SubscribeAudioAutoAndOnly = 2
    SubscribeCameraAutoAndOnly = 3


class VideoTrackType(Enum):
    VideoTrackCameraLarge = 0
    VideoTrackCameraSmall = 1
    VideoTrackCameraSuper = 2
    VideoTrackScreenshare = 3


class VideoFormat(Enum):
    VideoFormatNone = 0
    VideoFormatH264 = 1


class AudioFormat(Enum):
    AudioFormatNone = 0
    AudioFormatMixedPcm = 1
    AudioFormatPcmBeforMixing = 2


class VideoFrameType(Enum):
    VideoFrameH264 = 0     # YUV I420, image decoded
    VideoFrameH264Nalu = 1 # H264 Nalu, image before decoding


class AudioFrameType(Enum):
    AudioFrameRawPcm = 0
    AudioFrameAacAdts = 1


class AudioPcmFrame(object):
    frame_ms_:int     # 10ms
    channels_:int     # 1
    sample_bits_:int  # 16
    sample_rates_:int # 8k, 16k, 32k
    samples_:int
    streamIndex_:int # 0-默认音频流，1-第二音频流
    pcmBuf_:bytes
    pcmBufSize_:int

    def __init__(self, frame_ms, sample_rates, samples) -> None:
        self.frame_ms_ = frame_ms
        self.sample_rates_ = sample_rates
        self.samples_ = samples


class AudioAacFrame(object):
    frame_ms_:int     # 10ms
    channels_:int     # 1
    sample_rates_:int # 8k, 16k, 32k
    aacBuf_:bytes
    aacBufSize_:int


'''
 * @brief 音频数据源类型
'''
class AliRtcAudioSource(Enum):
    # 推流的音频数据
    AliRtcAudioSourcePub = 2
    # 推流和播放混音后的音频数据
    AliRtcAudioSourceMixedAll = 4

'''
 * @brief 音频采样率类型
'''
class AliRtcAudioSampleRate(Enum):
    # 8000采样率
    AliRtcAudioSampleRate_8000 = 0
    # @deprecated 11025采样率
    AliRtcAudioSampleRate_11025 = 1
    # 16000采样率
    AliRtcAudioSampleRate_16000 = 2
    # @deprecated 22050采样率
    AliRtcAudioSampleRate_22050 = 3
    # 32000采样率
    AliRtcAudioSampleRate_32000 = 4
    # 44100采样率
    AliRtcAudioSampleRate_44100 = 5
    # 48000采样率
    AliRtcAudioSampleRate_48000 = 6
    # 12000采样率
    AliRtcAudioSampleRate_12000 = 7
    # 24000采样率
    AliRtcAudioSampleRate_24000 = 8
    # 未知采样率
    AliRtcAudioSampleRate_Unknown = 100

'''
 * @brief 音频声道类型
'''
class AliRtcAudioNumChannelType(Enum):
    # 单声道
    AliRtcMonoAudio = 1
    # 双声道
    AliRtcStereoAudio = 2

'''
 * @brief 音频数据回调参数设置
'''
class AliRtcAudioFrameObserverConfig(object):
    # 回调音频采样率
    sampleRate:AliRtcAudioSampleRate
    # 回调音频声道数
    channels:AliRtcAudioNumChannelType
    def __init__(self) -> None:
        self.sampleRate = AliRtcAudioSampleRate.AliRtcAudioSampleRate_48000
        self.channels = AliRtcAudioNumChannelType.AliRtcMonoAudio


class GrtnNodeDelay(object):
    grtn_node_internal_rtt_half: int
    grtn_node_bef_pacer: int
    grtn_node_pacer_cost: int
    def __init__(self) -> None:
        self.grtn_node_internal_rtt_half = -1
        self.grtn_node_bef_pacer = -1
        self.grtn_node_pacer_cost = -1
    def __str__(self):
        return (f"GrtnNodeDelay(internal_rtt_half={self.grtn_node_internal_rtt_half}, "
                f"bef_pacer={self.grtn_node_bef_pacer}, "
                f"pacer_cost={self.grtn_node_pacer_cost})")

class AIAudioQuestionDelay(object):
    """
    ai 场景，
    客户端问题的最后一个音频 pcm 的采集时间戳。
    客户端提问，Linux SDK 拉流，Linux SDK 发送机器人回答时，
    ai_question_capture_timestamp 是客户端提问的时间戳。
    当取值为 0 的时候，意味着并不需要以下 ai 场景的延迟填充和发送。
    """
    ai_question_capture_timestamp: int
    ai_answer_capture_timestamp: int
    ai_question_agent_begin_timestamp: int
    ai_answer_agent_begin_timestamp: int
    # 提问的id
    ai_sentence_id: int
    """
    ai回环延迟相关，以下为linux 机器人回答的推流中需要设置， 对应的是客户端提问的链路延迟的转发
    客户端提问题的语音的上行延迟
    """
    ai_client_audio_pub_total_delay: int
    ai_client_audio_source_cost: int
    ai_client_audio_mixer_cost: int
    ai_client_audio_encoder_cost: int
    ai_client_audio_netsdk_thr_cost: int
    ai_client_audio_qos_thr_cost: int
    ai_client_audio_pacer_cost: int
    ai_client_up_half_rtt: int
    # 客户端上行 到 linux 端sub 期间 grtn的延迟
    ai_linux_grtn_node_delay: List[GrtnNodeDelay]
    ai_linux_down_half_rtt: int

    # linux 端 sub的延迟
    ai_linux_audio_sub_total_delay: int
    ai_linux_audio_receive_cost: int
    ai_linux_audio_neteq_cost: int
    ai_linux_audio_remote_source_cost: int
    ai_linux_audio_play_mixer_cost: int

    # ai agent 延迟
    ai_linux_audio_player_cost: int
    ai_agent_audio_asr_cost: int
    ai_agent_audio_llm_cost: int
    ai_agent_audio_tts_cost: int
    ai_agent_audio_total_cost: int

    # 音频处理延迟
    ai_linux_audio_process_smart_denoise_delay: int

    def __init__(self) -> None:
        self.ai_question_capture_timestamp = 0
        self.ai_answer_capture_timestamp = 0
        self.ai_question_agent_begin_timestamp = 0
        self.ai_answer_agent_begin_timestamp = 0
        self.ai_sentence_id = 0
        self.ai_client_audio_pub_total_delay = 0
        self.ai_client_audio_source_cost = 0
        self.ai_client_audio_mixer_cost = 0
        self.ai_client_audio_encoder_cost = 0
        self.ai_client_audio_netsdk_thr_cost = 0
        self.ai_client_audio_qos_thr_cost = 0
        self.ai_client_audio_pacer_cost = 0
        self.ai_client_up_half_rtt = 0
        self.ai_linux_grtn_node_delay = []
        self.ai_linux_down_half_rtt = 0
        self.ai_linux_audio_sub_total_delay = 0
        self.ai_linux_audio_receive_cost = 0
        self.ai_linux_audio_neteq_cost = 0
        self.ai_linux_audio_remote_source_cost = 0
        self.ai_linux_audio_play_mixer_cost = 0
        self.ai_linux_audio_player_cost = 0
        self.ai_agent_audio_asr_cost = 0
        self.ai_agent_audio_llm_cost = 0
        self.ai_agent_audio_tts_cost = 0
        self.ai_agent_audio_total_cost = 0
        self.ai_linux_audio_process_smart_denoise_delay = 0
    
    def copyFrom(self, src) -> None:
        if not isinstance(src, AIAudioQuestionDelay):
            return
        self.ai_sentence_id = src.ai_sentence_id
        self.ai_agent_audio_asr_cost = src.ai_agent_audio_asr_cost
        self.ai_agent_audio_llm_cost = src.ai_agent_audio_llm_cost
        self.ai_agent_audio_tts_cost = src.ai_agent_audio_tts_cost
        self.ai_agent_audio_total_cost = src.ai_agent_audio_total_cost
        self.ai_question_capture_timestamp = src.ai_question_capture_timestamp
        self.ai_answer_capture_timestamp = src.ai_answer_capture_timestamp
        self.ai_question_agent_begin_timestamp = src.ai_question_agent_begin_timestamp
        self.ai_answer_agent_begin_timestamp = src.ai_answer_agent_begin_timestamp

        self.ai_client_audio_pub_total_delay = src.ai_client_audio_pub_total_delay
        self.ai_client_audio_source_cost = src.ai_client_audio_source_cost
        self.ai_client_audio_mixer_cost = src.ai_client_audio_mixer_cost
        self.ai_client_audio_encoder_cost = src.ai_client_audio_encoder_cost
        self.ai_client_audio_netsdk_thr_cost = src.ai_client_audio_netsdk_thr_cost
        self.ai_client_audio_qos_thr_cost = src.ai_client_audio_qos_thr_cost
        self.ai_client_audio_pacer_cost = src.ai_client_audio_pacer_cost
        self.ai_client_up_half_rtt = src.ai_client_up_half_rtt
        self.ai_linux_down_half_rtt = src.ai_linux_down_half_rtt

        self.ai_linux_audio_sub_total_delay = src.ai_linux_audio_sub_total_delay
        self.ai_linux_audio_receive_cost = src.ai_linux_audio_receive_cost
        self.ai_linux_audio_neteq_cost = src.ai_linux_audio_neteq_cost
        self.ai_linux_audio_remote_source_cost = src.ai_linux_audio_remote_source_cost
        self.ai_linux_audio_play_mixer_cost = src.ai_linux_audio_play_mixer_cost
        self.ai_linux_audio_player_cost = src.ai_linux_audio_player_cost
        self.ai_linux_audio_process_smart_denoise_delay = src.ai_linux_audio_process_smart_denoise_delay

        self.ai_linux_grtn_node_delay.clear()
        for node in src.ai_linux_grtn_node_delay:
            new_node = GrtnNodeDelay()
            new_node.grtn_node_internal_rtt_half = node.grtn_node_internal_rtt_half
            new_node.grtn_node_bef_pacer = node.grtn_node_bef_pacer
            new_node.grtn_node_pacer_cost = node.grtn_node_pacer_cost
            self.ai_linux_grtn_node_delay.append(new_node)

    def __str__(self):
        return (f"AIAudioQuestionDelay(\n"
                f"  ai_question_capture_timestamp={self.ai_question_capture_timestamp}, \n"
                f"  ai_answer_capture_timestamp={self.ai_answer_capture_timestamp}, \n"
                f"  ai_question_agent_begin_timestamp={self.ai_question_agent_begin_timestamp}, \n"
                f"  ai_answer_agent_begin_timestamp={self.ai_answer_agent_begin_timestamp}, \n"
                f"  ai_sentence_id={self.ai_sentence_id}, \n"
                f"  ai_client_audio_pub_total_delay={self.ai_client_audio_pub_total_delay}, \n"
                f"  ai_client_audio_source_cost={self.ai_client_audio_source_cost}, \n"
                f"  ai_client_audio_mixer_cost={self.ai_client_audio_mixer_cost}, \n"
                f"  ai_client_audio_encoder_cost={self.ai_client_audio_encoder_cost}, \n"
                f"  ai_client_audio_netsdk_thr_cost={self.ai_client_audio_netsdk_thr_cost}, \n"
                f"  ai_client_audio_qos_thr_cost={self.ai_client_audio_qos_thr_cost}, \n"
                f"  ai_client_audio_pacer_cost={self.ai_client_audio_pacer_cost}, \n"
                f"  ai_client_up_half_rtt={self.ai_client_up_half_rtt}, \n"
                f"  ai_linux_grtn_node_delay={[str(node) for node in self.ai_linux_grtn_node_delay]}, \n"
                f"  ai_linux_down_half_rtt={self.ai_linux_down_half_rtt}, \n"
                f"  ai_linux_audio_sub_total_delay={self.ai_linux_audio_sub_total_delay}, \n"
                f"  ai_linux_audio_receive_cost={self.ai_linux_audio_receive_cost}, \n"
                f"  ai_linux_audio_neteq_cost={self.ai_linux_audio_neteq_cost}, \n"
                f"  ai_linux_audio_remote_source_cost={self.ai_linux_audio_remote_source_cost}, \n"
                f"  ai_linux_audio_play_mixer_cost={self.ai_linux_audio_play_mixer_cost}, \n"
                f"  ai_linux_audio_player_cost={self.ai_linux_audio_player_cost}, \n"
                f"  ai_agent_audio_asr_cost={self.ai_agent_audio_asr_cost}, \n"
                f"  ai_agent_audio_llm_cost={self.ai_agent_audio_llm_cost}, \n"
                f"  ai_agent_audio_tts_cost={self.ai_agent_audio_tts_cost}, \n"
                f"  ai_agent_audio_total_cost={self.ai_agent_audio_total_cost}, \n"
                f"  ai_linux_audio_process_smart_denoise_delay={self.ai_linux_audio_process_smart_denoise_delay} \n"
                f")")

class AIExtraDelay(object):
    avatar_render_cost: int

    def __init__(self):
        self.avatar_render_cost = 0
    
    def copyFrom(self, src) -> None:
        if not isinstance(src, AIExtraDelay):
            return
        self.avatar_render_cost = src.avatar_render_cost


class AudioFrame(object):
    type:AudioFrameType
    pcm:AudioPcmFrame
    aac:AudioAacFrame


class AliEngineVoiceEnvironment(Enum):
    AliEngineVoiceNoResult = -1
    AliEngineVoiceMonologue = 0
    AliEngineVoiceDialogue = 1
    AliEngineVoiceNoisy = 2
    AliEngineVoiceNoisyAndDialogue = 3


class AliEngineVoiceprintStatus(object):
    voiceprintEnable:bool
    voiceIsMainSpeaker:int
    voiceIsEnrolled:int
    voiceEnvironment:AliEngineVoiceEnvironment


class AudioTranscodingCodec(Enum):
    AudioTranscodingCodecPcm = 0
    AudioTranscodingCodecAac = 1
    AudioTranscodingCodecBothPcmAndAac = 2
    AudioTranscodingCodecMax = 3


class VideoTranscodingCodec(Enum):
    VideoTranscodingCodecYuv = 0
    VideoTranscodingCodecH264 = 1
    VideoTranscodingCodecBothYuvAndH264 = 2
    VideoTranscodingCodecMax = 3

class AliEngineConnectionStatus(Enum):
    AliEngineConnectionStatusInit = 0
    AliEngineConnectionStatusDisconnected = 1
    AliEngineConnectionStatusConnecting = 2
    AliEngineConnectionStatusConnected = 3
    AliEngineConnectionStatusReconnecting = 4
    AliEngineConnectionStatusFailed = 5

class AliEngineConnectionStatusChangeReason(Enum):
    AliEngineConnectionChangedDummyReason = 0
    AliEngineConnectionMediaPathChanged = 1
    AliEngineConnectionSignalingHeartbeatTimeout = 2
    AliEngineConnectionSignalingHeartbeatAlive = 3
    AliEngineConnectionSignalingHttpDnsResolved = 4
    AliEngineConnectionSignalingHttpDnsFailure = 5
    AliEngineConnectionSignalingGslbFailure = 6
    AliEngineConnectionSignalingGslbSucccess = 7
    AliEngineConnectionSignalingJoinRoomFailure = 8
    AliEngineConnectionSignalingJoinRoomSuccess = 9
    AliEngineConnectionSignalingLeaveRoom = 10
    AliEngineConnectionSignalingConnecting = 11
    AliEngineConnectionChangedNetworkInterrupted = 12

class VideoH264Frame(object):
    frame_ms_:int
    frame_num_:int  
    buf_:bytes
    bufSize_:int
    width:int
    height:int
    qp:int
    mv_infos:List[Tuple[int, int]]
    slice_type:int


class VideoFrame(object):
    track:VideoTrackType # camera or screen
    type:VideoFrameType  # yuv or h264 nalu
    frame: VideoH264Frame


'''
 * 水印设置
 * x, y, width, height分别为水印的位置和大小
 * alpha为水印的透明度
 * normalized表示水印的位置和大小是否是归一化的。
 * 若normalize=true，x，y，width取值范围是0～1的浮点数；height不用设置，SDK会根据水印图片的宽高比计算出合适的高度；
 * 若normalize=false，x，y，width，height都为绝对的像素值；
'''
class WaterMarkConfig(object):
    x = 0.0
    y = 0.0
    width = 0.0
    height = 0.0
    alpha = 1.0
    normalized = False


class MediaPlayerState(Enum):
    MediaPlayerStateIdle = 0
    MediaPlayerStateInited = 1
    MediaPlayerStatePrepareing = 2
    MediaPlayerStatePrepared = 3
    MediaPlayerStateStarted = 4
    MediaPlayerStatePaused = 5
    MediaPlayerStateCompleted = 6
    MediaPlayerStateStopped = 7
    MediaPlayerStateError = 8
    MediaPlayerStateEnd = 9


class MediaPlayerEvent(Enum):
    MediaPlayerEventBufferingStart = 100
    MediaPlayerEventBufferingEnd = 200
    MediaPlayerEventSeekComplete = 300
    MediaPlayerEventErrorOccured = 900


class AliEnginePublishState(Enum):
    # 初始状态
    AliEngineStatsPublishIdle = 0
    # 未推流
    AliEngineStatsNoPublish = 1
    # 推流中
    AliEngineStatsPublishing = 2
    # 已推流
    AliEngineStatsPublished = 3


class AliEngineSubscribeState(Enum):
    # 初始状态
    AliEngineStatsSubscribeIdle = 0
    # 未订阅
    AliEngineStatsNoSubscribe = 1  
    # 订阅中  
    AliEngineStatsSubscribing = 2  
    # 已订阅  
    AliEngineStatsSubscribed = 3     


class AliEngineVideoStreamType(Enum):
    # 无，在OnSubscribeStreamTypeChanged回调表示当前未订阅
    AliEngineVideoStreamTypeNone = 0
    # 高码率，高分辨率流（大流）
    AliEngineVideoStreamTypeHigh = 1
    # 低码率，低分辨率流（小流） 
    AliEngineVideoStreamTypeLow = 2


class AliEngineMuteLocalAudioMode(Enum):
    # 默认模式(静音全部，包括麦克风及外部输入音频) 
    AliEngineMuteLocalAudioModeDefault      = 0
    # 只静音麦克风
    AliEngineMuteLocalAudioModeMuteOnlyMic  = 1
    # 静音全部(包括麦克风及外部输入音频) 
    AliEngineMuteLocalAudioModeMuteAll      = 2


# 用户角色类型
class AliEngineClientRole(Enum):
    # 互动角色 
    AliEngineClientRoleInteractive = 0
    # 观众角色
    AliEngineClientRoleLive = 1

    
class JoinChannelConfig(object):
    channelProfile: ChannelProfile    
    isAudioOnly:bool
    publishAvsyncMode: PublishAvsyncMode
    publishAvsyncWithPtsMaxAudioCacheSize: int
    publishAvsyncWithPtsMaxVideoCacheSize: int
    publishMode: PublishMode
    subscribeMode: SubscribeMode
    subscribeVideoFormat: VideoFormat
    subscribeAudioFormat: AudioFormat
    enableTtsCallback: bool
    
    def __init__(self) -> None:
        self.channelProfile = ChannelProfile.ChannelProfileInteractiveLive
        self.isAudioOnly = False
        self.publishAvsyncMode = PublishAvsyncMode.PublishAvsyncNoDelay
        self.publishAvsyncWithPtsMaxAudioCacheSize = 300
        self.publishAvsyncWithPtsMaxVideoCacheSize = 100
        self.publishMode = PublishMode.PublishManually
        self.subscribeMode = SubscribeMode.SubscribeManually
        self.subscribeVideoFormat = VideoFormat.VideoFormatH264
        self.subscribeAudioFormat = AudioFormat.AudioFormatMixedPcm
        self.enableTtsCallback = False


class AliCapabilityProfile(Enum):
    AliCapabilityProfileDefault = 0
    AliCapabilityProfileAiHuman = 1
    AliCapabilityProfileAiRobot = 2

class AliEngineUserParam(object):
    channelId: str
    userId: str
    userName: str
    capabilityProfile: AliCapabilityProfile
    useVoip: bool
    def __init__(self) -> None:
        self.channelId = ""
        self.userId = ""
        self.userName = ""
        self.capabilityProfile = AliCapabilityProfile.AliCapabilityProfileAiRobot
        self.useVoip = False

    def __init__(self, channelId:str, userId:str) -> None:
        self.channelId = channelId
        self.userId = userId
        self.userName = userId
        self.capabilityProfile = AliCapabilityProfile.AliCapabilityProfileAiRobot
        self.useVoip = False

    def __init__(self, channelId:str, userId:str, userName:str, capabilityProfile:AliCapabilityProfile) -> None:
        self.channelId = channelId
        self.userId = userId
        self.userName = userName
        self.capabilityProfile = capabilityProfile
        self.useVoip = False



class AuthInfo(object):
    channel: str
    userid: str
    username: str
    appid: str
    token: str
    timestamp: int
    nonce: str
    role: str
    
    def __init__(self) -> None:
        self.channel = None
        self.userid = None
        self.username = None
        self.appid = None
        self.token = None
        self.timestamp = 0
        self.nonce = None
        self.role = None

# 伴奏控制指令模式
class AliEngineDataMsgType(Enum):
    AliEngineDataChannelNone = 0
    AliEngineDataChannelProgress = 1
    AliEngineDataChannelCustom = 2


# 伴奏控制消息
class AliEngineDataChannelMsg(object):
    type: AliEngineDataMsgType    
    networkTime: int
    progress: int
    data: bytes
    dataLen: int

class AudioObserverManualConfig(object):
    enableDenoise: bool
    enableVoiceprintRecognize: bool
    enableAsrCallback: bool
    enableTtsVad: bool
    isBindPublishLoopDelay: bool
    def __init__(self) -> None:
        self.enableDenoise = False
        self.enableVoiceprintRecognize = False
        self.enableAsrCallback = False
        self.enableTtsVad = False
        self.isBindPublishLoopDelay = False

class BufferDataType(Enum):
    AudioOnly = 0
    VideoOnly = 1
    BothAudioAndVideo = 2

'''
音频音效播放配置
'''
class AliRtcAudioEffectConfig(object):
    # 是否推流，默认值：False
    needPublish: bool
    # 循环次数，可以设置-1(无限循环)或者>0的正整数次，其他值无效，默认值：-1
    loopCycles: int
    # 起播位置，单位：ms，默认值：0
    startPosMs: int
    # 推流音量，取值范围[0-100]，默认值：50
    publishVolume: int
    # 播放音量，取值范围[0-100]，默认值：50
    playoutVolume: int
    def __init__(self) -> None:
        self.needPublish = False
        self.loopCycles = -1
        self.startPosMs = 0
        self.publishVolume = 50
        self.playoutVolume = 50

'''
通话呼出所需信息
'''
class AliEngineDialInfo(object):
    sipDomain: str # Sip中继服务器域名
    sipToken: str # Sip鉴权token
    callerNumber: str # 主叫号码
    calleeNumber: str # 被叫号码
    calleePrefix: str # 被叫前缀
    localPublicIp: str # 本地公网ip,为空字符串,内部则取eth0的ip代替
    minPort: int # 最小端口
    maxPort: int # 最大端口
    sipHeaderExtra: str # Sip头扩展，json字符串
    def __init__(self) -> None:
        self.sipDomain = None
        self.sipToken = None
        self.callerNumber = None
        self.calleeNumber = None
        self.calleePrefix = None
        self.localPublicIp = None
        self.minPort = 0
        self.maxPort = 0
        self.sipHeaderExtra = None

'''
媒体配置信息
'''
class AliEngineVoipConfig(object):
    audioCodec: str # 音频编码类型
    audioSampleRate: int # 音频采样率
    
    def __init__(self) -> None:
        self.audioCodec = "PCMA"
        self.audioSampleRate = 8000

'''
通话呼入返回信息
'''
class AliEnginePickupIncomingCallInfo(object):
    sdpInfo: str # 响应的sdp信息
    result: int # 响应状态码

    def __init__(self) -> None:
        self.sdpInfo = None
        self.result = 0

'''
通话呼入所需信息
'''
class AliEngineIncomingCallInfo(object):
    sipDomain: str # Sip中继服务器域名
    callerNumber: str # 主叫号码
    calleeNumber: str # 被叫号码
    remotePublicIp: str # 呼入方的IP
    remoteAudioPort: int # 呼入方的端口
    minPort: int # 本地端口最小分配端口
    maxPort: int # 本地端口最大分配端口
    inviteSDPInfo: str # 呼入的sdp内容，如果传入此参数，则remotePublicIp，remoteAudioPort，audioCodec和audioSampleRate会被替换成sdp信息里的设置

    def __init__(self) -> None:
        self.sipDomain = None
        self.callerNumber = None
        self.calleeNumber = None
        self.remotePublicIp = None
        self.remoteAudioPort = 0
        self.minPort = 0
        self.maxPort = 0
        self.inviteSDPInfo = None

# ================== 统计信息类 ==================
class AliEngineStats(object):
    def __init__(self):
        self.availableSendBitrate: int = 0
        self.sentKBitrate: int = 0
        self.rcvdKBitrate: int = 0
        self.sentBytes: int = 0
        self.rcvdBytes: int = 0
        self.videoRcvdKBitrate: int = 0
        self.videoSentKBitrate: int = 0
        self.systemCpu: int = 0
        self.appCpu: int = 0
        self.callDuration: int = 0
        self.sentLossRate: int = 0
        self.sentLossPkts: int = 0
        self.sentExpectedPkts: int = 0
        self.rcvdLossRate: int = 0
        self.rcvdLossPkts: int = 0
        self.rcvdExpectedPkts: int = 0
        self.lastmileDelay: int = 0


class AliEngineLocalVideoStats(object):
    def __init__(self):
        self.track: VideoTrack = VideoTrack.VideoTrackNo
        self.targetEncodeBitrate: int = 0
        self.actualEncodeBitrate: int = 0
        self.sentBitrate: int = 0
        self.sentBitrateKbps: int = 0
        self.sentFps: int = 0
        self.encodeFps: int = 0
        self.captureFps: int = 0
        self.avgQpPerSec: int = 0
        self.rtt: int = 0
        self.sendBytes: int = 0
        self.renderFps: int = 0


class AliEngineRemoteVideoStats(object):
    def __init__(self):
        self.userId: str = ""
        self.track: VideoTrack = VideoTrack.VideoTrackNo
        self.width: int = 0
        self.height: int = 0
        self.decodeFps: int = 0
        self.renderFps: int = 0
        self.frozenTimes: int = 0
        self.videoTotalFrozenTime: int = 0
        self.videoTotalFrozenRate: int = 0
        self.rtpCount: int = 0
        self.rtpLoss: int = 0
        self.rtt: int = 0
        self.recvBitrate: int = 0
        self.rcvdBitrateKbps: int = 0
        self.e2eDelay: int = 0
        self.frameLoss: int = 0
        self.pullBytes: int = 0
        self.packetLossRate: float = 0.0


class AliEngineLocalAudioStats(object):
    def __init__(self):
        self.scene: AudioSceneMode = AudioSceneMode.DefaultMode
        self.track: AudioTrack = AudioTrack.AudioTrackNo
        self.sentBitrate: int = 0
        self.sentBitrateKbps: int = 0
        self.sentSamplerate: int = 0
        self.numChannel: int = 0
        self.inputLevel: int = 0
        self.rtt: int = 0
        self.sendBytes: int = 0


class AliEngineRemoteAudioStats(object):
    def __init__(self):
        self.userId: str = ""
        self.track: AudioTrack = AudioTrack.AudioTrackNo
        self.quality: int = 0
        self.networkTransportDelay: int = 0
        self.jitterBufferDelay: int = 0
        self.audioLossRate: float = 0.0
        self.rcvdBitrate: int = 0
        self.rcvdBitrateKbps: int = 0
        self.totalFrozenTimes: int = 0
        self.audioTotalFrozenTime: int = 0   # 音频播放的累计卡顿时长，单位ms
        self.audioTotalFrozenRate: int = 0   # 音频播放卡顿率，单位%
        self.rtt: int = 0
        self.e2eDelay: int = 0
        self.ai_e2eDelay: int = 0
        self.pullBytes: int = 0