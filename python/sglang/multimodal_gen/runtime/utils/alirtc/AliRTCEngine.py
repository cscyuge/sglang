from AliRTCLinuxSdkDefine import *
from abc import ABC, abstractmethod

import asyncio
import time

class EngineEventHandlerInterface(object):
    
    '''
     * @brief 如果engine出现error，通过这个消息通知业务层
     * @param error_code  Error type
    '''
    def OnError(self, error_code:ERROR_CODE) -> None:
        pass

    '''
     * @brief 如果engine出现warning，通过这个消息通知业务层
     * @param warning_code  Warning type
    '''
    def OnWarning(self, warning_code:WARNNING_CODE) -> None:
        pass
   
    '''
     * @brief 加入频道结果
     * @details 当应用调用 {@link JoinChannel} 方法时，该回调表示成功/失败加入频道
     * @param result 加入频道结果，成功返回0，失败返回错误码
    '''
    def OnJoinChannelResult(self, result:int, channel:str, userId:str) -> None:
        pass
   
    '''
     * @brief 可以开始发送data channel消息回调
     * @param uid 用户
     * @note 收到此回调后，可以开始发送data channel消息
    '''
    def OnRemoteUserSubscribedDataChannel(self, uid:str) -> None:
        pass

    '''
     * @brief 获得dataChannel远端数据
     * @param msg 远端传来的消息
    '''
    def OnDataChannelMsg(self, uid:str, msg:AliEngineDataChannelMsg) -> None:
        pass

    '''
     * @brief 离开频道结果
     * @details 应用调用 {@link LeaveChannel} 方法时，该回调表示成功/失败离开频道，回调将会返回离会的result,如果 {@link LeaveChannel} 后直接 {@link Destory} SDK，将不会收到此回调
     * @param result 离开频道结果，成功返回0，失败返回错误码
    '''
    def OnLeaveChannelResult(self, result:int) -> None:
        pass

    '''
     * @brief 远端用户（通信模式）/（互动模式，主播角色）加入频道回调
     * @details 该回调在以下场景会被触发
     * - 通信模式：远端用户加入频道会触发该回调，如果当前用户在加入频道前已有其他用户在频道中，当前用户加入频道后也会收到已加入频道用户的回调
     * - 互动模式：
     *   - 远端主播角色用户加入频道会触发该回调，如果当前用户在加入频道前已有其他主播在频道中，当前用户加入频道后也会收到已加入频道主播的回调
     *   - 远端观众角色调用 {@link SetClientRole} 切换为主播角色 {@link AliEngineClientRoleInteractive}，会触发该回调
     *
     * @param uid 用户ID 从App server分配的唯一标示符
     * @note 互动模式下回调行为
     * - 主播间可以互相收到加入频道回调
     * - 观众可以收到主播加入频道回调
     * - 主播无法收到观众加入频道回调
    '''
    def OnRemoteUserOnLineNotify(self, uid:str) -> None:
        pass
   
    '''
     * @brief 远端用户（通信模式）/（互动模式，主播角色）离开频道回调
     * @details 该回调在以下场景会被触发
     * - 通信模式：远端用户离开频道会触发该回调
     * - 互动模式：
     *   - 远端主播角色{@link AliEngineClientRoleInteractive}离开频道
     *   - 远端主播切换调用 {@link SetClientRole} 切换为观众角色{@link AliEngineClientRoleLive}，会触发该回调
     * - 通信模式和互动模式主播角色情况下，当长时间收不到远端用户数据，超时掉线时，会触发该回调
     *
     * @param uid 用户ID 从App server分配的唯一标示符
    '''
    def OnRemoteUserOffLineNotify(self, uid:str) -> None:
        pass

    '''
     * @brief 远端用户的音视频流发生变化回调
     * @details 该回调在以下场景会被触发
     * - 当远端用户从未推流变更为推流（包括音频和视频）
     * - 当远端用户从已推流变更为未推流（包括音频和视频）
     * - 互动模式下，调用 {@link SetClientRole} 切换为主播角色 {@link AliEngineClientRoleInteractive}，同时设置了推流时，会触发该回调
     * @param uid userId，从App server分配的唯一标示符
     * @param audioTrack 音频流类型，详见 {@link AliEngineAudioTrack}
     * @param videoTrack 视频流类型，详见 {@link AliEngineVideoTrack}
     * @note 该回调仅在通信模式用户和互动模式下的主播角色才会触发
    '''
    def OnRemoteTrackAvailableNotify(self, uid:str, audioTrack:AudioTrack, videoTrack:VideoTrack) -> None:
        pass

    '''
     * @brief 音频订阅情况变更回调
     * @param uid userId，从App server分配的唯一标示符
     * @param oldState 之前的订阅状态，详见 {@link AliRTCSdk::AliEngineSubscribeState}
     * @param newState 当前的订阅状态，详见 {@link AliRTCSdk::AliEngineSubscribeState}
     * @param elapseSinceLastState 两次状态变更时间间隔(毫秒)
     * @param channel 当前频道id
    '''
    def OnAudioSubscribeStateChanged(self, uid:str, oldState:AliEngineSubscribeState, newState:AliEngineSubscribeState,
                                     elapseSinceLastState:int, channel:str) -> None:
        pass

    '''
     * @brief 相机流订阅情况变更回调
     * @param uid userId，从App server分配的唯一标示符
     * @param oldState 之前的订阅状态，详见 {@link AliRTCSdk::AliEngineSubscribeState}
     * @param newState 当前的订阅状态，详见 {@link AliRTCSdk::AliEngineSubscribeState}
     * @param elapseSinceLastState 两次状态变更时间间隔(毫秒)
     * @param channel 当前频道id
    '''
    def OnVideoSubscribeStateChanged(self, uid:str, oldState:AliEngineSubscribeState, newState:AliEngineSubscribeState,
                                     elapseSinceLastState:int, channel:str) -> None:
        pass

    '''
     * @brief 大小订阅情况变更回调
     * @param uid userId，从App server分配的唯一标示符
     * @param oldStreamType 之前的订阅的大小流类型，详见 {@link AliRTCSdk::AliEngineVideoStreamType}
     * @param newStreamType 当前的订阅的大小流类型，详见 {@link AliRTCSdk::AliEngineVideoStreamType}
     * @param elapseSinceLastState 大小流类型变更时间间隔(毫秒)
     * @param channel 当前频道id
    '''
    def OnSubscribeStreamTypeChanged(self, uid:str, oldStreamType:AliEngineVideoStreamType, newStreamType:AliEngineVideoStreamType,
                                     elapseSinceLastState:int, channel:str) -> None:
        pass
    
    '''
     * @brief 屏幕分享流订阅情况变更回调
     * @param uid userId，从App server分配的唯一标示符
     * @param oldState 之前的订阅状态，详见 {@link AliRTCSdk::AliEngineSubscribeState}
     * @param newState 当前的订阅状态，详见 {@link AliRTCSdk::AliEngineSubscribeState}
     * @param elapseSinceLastState 两次状态变更时间间隔(毫秒)
     * @param channel 当前频道id
    '''
    def OnScreenShareSubscribeStateChanged(self, uid:str, oldState:AliEngineSubscribeState, newState:AliEngineSubscribeState,
                                           elapseSinceLastState:int, channel:str) -> None:
        pass

    '''
     * @brief 屏幕分享推流变更回调
     * @param oldState 之前的推流状态
     * @param newState 当前的推流状态
     * @param elapseSinceLastState 状态变更时间间隔
     * @param channel 当前频道id
    '''
    def OnScreenSharePublishStateChanged(self, oldState:AliEnginePublishState, newState:AliEnginePublishState,
                                         elapseSinceLastState:int, channel:str) -> None:
        pass

    '''
     * @brief 次要流推流变更回调
     * @param oldState 之前的推流状态
     * @param newState 当前的推流状态
     * @param elapseSinceLastState 状态变更时间间隔
     * @param channel 当前频道id
    '''
    def OnDualStreamPublishStateChanged(self, oldState:AliEnginePublishState, newState:AliEnginePublishState,
                                        elapseSinceLastState:int, channel:str) -> None:
        pass

    '''
     * @brief 视频推流变更回调
     * @param oldState 之前的推流状态
     * @param newState 当前的推流状态
     * @param elapseSinceLastState 状态变更时间间隔
     * @param channel 当前频道id
    '''
    def OnVideoPublishStateChanged(self, oldState:AliEnginePublishState, newState:AliEnginePublishState,
                                   elapseSinceLastState:int, channel:str) -> None:
        pass
    
    '''
     * @brief 音频推流变更回调
     * @param oldState 之前的推流状态
     * @param newState 当前的推流状态
     * @param elapseSinceLastState 状态变更时间间隔
     * @param channel 当前频道id
    '''
    def OnAudioPublishStateChanged(self, oldState:AliEnginePublishState, newState:AliEnginePublishState,
                                   elapseSinceLastState:int, channel:str) -> None:
        pass   
    
    '''
     * @brief 网络状态变化回调
     * @param status 当前网络状态
     * @param reason 状态变化原因
    '''
    def OnConnectionStatusChanged(self, status:AliEngineConnectionStatus, reason:AliEngineConnectionStatusChangeReason) -> None:
        pass

    '''
     * @brief 混音前每一路远端用户的音频数据回调
     * @details 来自远端单个用户的音频数据 对应AliRTCSdk::Linux::AudioFormatPcmBeforMixing
     * @param frame 音频数据，详见{@link AliRTCSdk::Linux::AudioFrame}
    '''
    def OnSubscribeAudioFrame(self, uid:str, frame:AudioFrame) -> None:
        pass

    '''
     * @brief 本地订阅音频数据回调
     * @details 远端所有用户混音后待播放的音频数据 对应AliRTCSdk::Linux::AudioFormatMixedPcm
     * @param frame 音频数据，详见{@link AliRTCSdk::Linux::AudioFrame}
    '''
    def OnSubscribeMixAudioFrame(self, frame:AudioFrame) -> None:
        pass

    '''
     * @brief 所有用户音频数据混合后回调 
     * @param frame 音频数据，详见{@link AliRTCSdk::Linux::AudioFrame}
     * @note 请不要在此回调函数中做任何耗时操作，否则可能导致声音异常
    '''
    def OnMixedAllAudioFrame(self, frame:AudioFrame) -> None:
        pass

    '''
     * @brief 推流数据回调
     * @details 请不要在此回调函数中做任何耗时操作，否则可能导致声音异常
     * @param frame 音频数据，详见{@link AliRTCSdk::Linux::AudioFrame}
    '''
    def OnPublishAudioFrame(self, frame:AudioFrame) -> None:
        pass

    '''
     * @brief Qos参数发生变化通知
     * @param trackType 变化视频track
     * @param paramter qos参数结构体
    '''
    def OnRequestVideoExternalEncoderParameter(self, trackType:VideoTrack, paramter:AliEngineVideoExternalEncoderParameter) -> None:
        pass

    '''
    * @brief Qos请求帧类型发生变化通知
    * @param trackType 变化视频track
    * @param frameType 请求参考帧类型
    * @param dropFrame 是否丢帧
    '''
    def OnRequestVideoExternalEncoderFrame(self, trackType:VideoTrack, frameType:AliEngineVideoEncodedFrameType, dropFrame:bool) -> None:
        pass

    '''
     * @brief 订阅的远端视频数据回调
     * @param uid 用户ID
     * @param frame 视频裸数据
     * @return
    '''
    def OnRemoteVideoSample(self, uid:str, frame:VideoFrame) -> None:
        pass

    '''
     * @brief 订阅的远端视频，解码前数据回调
     * @param uid 用户ID
     * @param frame 视频解码前数据，Nalu
     * @return
    '''
    def OnRemoteVideoEncodedSample(self, uid:str, frame:VideoFrame) -> None:
        pass

    '''
     * @brief 收到媒体扩展信息回调
     * @param uid 发送用户userId
     * @param message 扩展信息内容
     * @param size 扩展信息长度
     * @note 当一端通过 {@link SendMediaExtensionMsg} 发送信息后，其他端通过该回调接收数据
    '''
    def OnMediaExtensionMsgReceived(self, userid:str, message:bytes, size:int) -> None:
        pass

    '''
     * @brief 当用户角色发生变化时通知
     * @param oldRole 变化前角色类型，参考{@link AliEngineClientRole}
     * @param newRole 变化后角色类型，参考{@link AliEngineClientRole}
     * @note 调用{@link setClientRole}方法切换角色成功时触发此回调
    '''
    def OnUpdateRoleNotify(self, oldRole:AliEngineClientRole, newRole:AliEngineClientRole) -> None:
        pass

    '''
     * @brief 音频存档定制能力，分流音频转码AAC输出
     * @param uid 远端用户ID
     * @param frame 音频数据，详见{@link AliRTCSdk::Linux::AudioFrame}
    '''
    def OnSubscribeAudioAac(self, uid:str, frame:AudioFrame) -> None:
        pass

    '''
     * @brief 音频存档定制能力，合流音频转码AAC输出
     * @param frame 音频数据，详见{@link AliRTCSdk::Linux::AudioFrame}
    '''
    def OnSubscribeMixedAudioAac(self, frame:AudioFrame) -> None:
        pass

    '''
     * @brief 非匀速情况下推送音频数据过快，SDK反馈状态
     * @param isFull SDK音频缓存是否已满，若已满请稍等些许再推音频数据
    '''
    def OnPushAudioFrameBufferFull(self, isFull:bool) -> None:
        pass

    '''
     * @brief 非匀速情况下推送视频数据过快，SDK反馈状态
     * @param isFull SDK视频缓存是否已满，若已满请稍等些许再推视频数据
    '''
    def OnPushVideoFrameBufferFull(self, isFull:bool) -> None:
        pass

    '''
     * @brief 获取剩余（未消费）缓存数据大小
     * @param bufferSize 缓存大小
    '''
    def OnGetRemainDataBufferSize(self, audioBufferSize:int, videoBufferSize:int) -> None:
        pass

    '''
     * @brief (AI场景定制能力) 回调主讲人状态
     * @param startTime 时间范围起始时间
     * @param endTime 时间范围结束时间
     * @param status 时间范围内说话方是否为主讲人的状态
    '''
    def OnGetAudioIsMainSpeaker(self, startTime:int, endTime:int, status:AliEngineVoiceprintStatus) -> None:
        pass

    '''
     * @brief 获取AIVad识别结果回调
     * @param startTime asr识别开始时间
     * @param endTime asr识别结束时间
     * @param status 0:非vip，1:vip
    '''
    def OnGetAIVadInfo(self, startTime:int, endTime:int, status:int) -> None:
        pass

    '''
     * @brief 获取时延信息加密结果回调
     * @param delay 被加密后的时延信息
    '''
    def OnEncryptQuestionDelayInfo(self, delay:str) -> None:
        pass

    '''
     * @brief 回调自动提取的声纹向量
     * @param 算法模块自动提取的声纹向量
    '''
    def OnVoiceprintVector(self, vector:bytes) -> None:
        pass

    '''
     * @brief AI场景，回调远端用户是否已准备好接收欢迎语
     * @param isReady 是否已准备好
    '''
    def OnGreetReady(self, isReady:bool) -> None:
        pass
    
    '''
     * @brief AI场景，回调远端用户的设备Id，用以声纹降噪
     * @param uid 远端用户Id
     * @param deviceId 远端设备Id
    '''
    def OnRemoteDeviceId(self, uid:str, deviceId:str) -> None:
        pass

    '''
     * @brief AI场景，在远端用户说完一句话后触发回调，用于记录ASR时间
     * @param uid 远端用户Id
     * @param isEnd 是否为一句话结束
    '''
    def OnRemoteSentenceState(self, uid:str, isEnd:bool, delayInfo:AIAudioQuestionDelay) -> None:
        pass

    '''
     * @brief AI场景，在本地用户开始说一句话时触发回调，用于记录TTS播放时间，以及获取底层感知的AI agent耗时
     * @param uid 对应统计耗时的远端用户Id
     * @param isBegin 是否为一句话开始
     * @param delayInfo 统计到的耗时
    '''
    def OnLocalSentenceState(self, uid:str, isBegin:bool, delayInfo:AIAudioQuestionDelay) -> None:
        pass

    '''
     * @brief 音频帧推流开始
     * @param sentenceID 回答的轮次ID
     * @param sequenceID 回答的句子ID
    '''
    def OnPushAudioFrameBegin(self, sentenceID:int, sequenceID:int) -> None:
        pass
    
    '''
     * @brief 音频帧推流结束
     * @param sentenceID 回答的轮次ID
     * @param sequenceID 回答的句子ID
    '''
    def OnPushAudioFrameEnd(self, sentenceID:int, sequenceID:int) -> None:
        pass

    '''
     * @brief 本地音效播放结束回调
     * @param soundId 用户给该音效文件分配的唯一ID
    '''
    def OnAudioEffectFinished(self, soundId:int) -> None:
        pass

    '''
     * @brief 音频首包接收回调
     * @details 在接收到远端首个音频数据包时触发此回调
     * @param uid 远端用户ID，从App server分配的唯一标识符
     * @param timeCost 接收耗时，从入会开始到音频首包接收到的耗时
    '''
    def OnFirstAudioPacketReceived(self, uid:str, timeCost:int) -> None:
        pass

    '''
     * @brief 在PSTN网络远端挂断通话时候回调
    '''
    def OnRemoteHangUp(self) -> None:
        pass

    '''
     * @brief 在PSTN网络本端主动挂断通话时回调
     * @param code 挂断原因/状态
    '''
    def OnHangUpResult(self, code:int) -> None:
        pass

    '''
     * @brief Voip通话质量回调
     * @param code 通话质量标识
    '''
    def OnVoipQuality(self, code:int) -> None:
        pass

    '''
     * @brief 在PSTN网络呼叫远端 返回消息时调用
     * @param code 返回码=0 是成功，其他错误
    '''
    def OnDialResult(self, code:int) -> None:
        pass

    '''
     * @brief 在PSTN网络呼叫远端 更新协商后返回消息时调用
     * @param code 返回码=0 是成功，其他错误
    '''
    def OnDialUpdateResult(self, code:int) -> None:
        pass

    '''
     * @brief 在PSTN网络呼叫远端 接通前的消息回调
     * @param code 状态码
    '''
    def OnDialStateChange(self, code:int) -> None:
        pass

    '''
     * @brief 在PSTN网络接收呼入 受理成功时回调
     * @param info 回调信息
    '''
    def OnPickupIncomingCall(self, info:AliEnginePickupIncomingCallInfo) -> None:
        pass

    '''
     * @brief 在PSTN网络接收呼入 媒体传输成功/失败时调用
     * @param code 状态码
    '''
    def OnConnectIncomingCallResult(self, code:int) -> None:
        pass

    '''
     * @brief 在PSTN网络接收呼入 媒体传输成功/失败时调用
     * @param code 状态码
    '''
    def OnDisConnectIncomingCallResult(self, code:int) -> None:
        pass

    '''
     * @brief Voip电话按键信号
     * @param eventStr 事件编号字符串
     * @param eventCode 事件编号：0=DTMF 0, 1=1, ..., 11=*, 12=#, 16=A~19=D
     * @param endFlag 1 表示这是该事件的最后一个包
     * @param volume 相对音频音量（dB），通常 0–63，一般设为 0
     * @param timestampIncrement 自上一个事件包以来时间戳增量（单位同 RTP 时间戳）
     * @param duration 事件持续时间（单位：采样点）
    '''
    def OnVoipTelephoneEvent(self, eventStr:str, eventCode:int, endFlag:int, volume:int, timestampIncrement:int, duration:int) -> None:
        pass
        
    '''
     * @brief 当前会话统计信息回调
     * @param stats 会话统计信息
     * @note SDK每两秒触发一次此统计信息回调
    '''
    def OnStats(self, stats:AliEngineStats) -> None:
        pass

    '''
     * @brief 本地视频统计信息
     * @param localVideoStats 本地视频统计信息
     * @note SDK每两秒触发一次此统计信息回调
    '''
    def OnLocalVideoStats(self, localVideoStats:AliEngineLocalVideoStats) -> None:
        pass

    '''
     * @brief 远端视频统计信息
     * @param remoteVideoStats 远端视频统计信息
     * @note SDK每两秒触发一次此统计信息回调
    '''
    def OnRemoteVideoStats(self, remoteVideoStats:AliEngineRemoteVideoStats) -> None:
        pass
    
    '''
     * @brief 本地音频统计信息
     * @param localAudioStats 本地视频统计信息
     * @note SDK每两秒触发一次此统计信息回调
    '''
    def OnLocalAudioStats(self, localAudioStats:AliEngineLocalAudioStats) -> None:
        pass
    
    '''
     * @brief 远端音频统计信息
     * @param remoteAudioStats 远端视频统计信息
     * @note SDK每两秒触发一次此统计信息回调
    '''
    def OnRemoteAudioStats(self, remoteAudioStats:AliEngineRemoteAudioStats) -> None:
        pass

    '''
     * @brief Profile统计信息上报
     * @param type Profile类型 1-视频，10-音频，11-AI回环
     * @param profileStats 统计信息，json格式
    '''
    def OnProfileStats(self, type:int, profileStats:str) -> None:
        pass

class AliRTCEngineInterface(ABC):
    '''
     * @brief 销毁AliRTCEngine的实例
    '''
    @abstractmethod
    def Release() -> int:
        pass
  
    '''
     * @brief 获取事件回调句柄
    '''
    @abstractmethod
    def GetEventHandler() -> EngineEventHandlerInterface:
        pass

    '''
     * @brief 加入频道，兼容多参数入会和单参数入会两种模式
     * 详见{@link https://help.aliyun.com/zh/live/user-guide/token-based-authentication?spm=a2c4g.11186623.0.0.315971e0tmFEgF}
     * @param authInfo    认证信息，从App Server获取。
     * @param config      joinChannel时的设置项
    '''
    @abstractmethod
    def JoinChannelFromServer(authInfo: AuthInfo, config: JoinChannelConfig) -> int:
        pass
        
    '''
     * @brief 单参数加入频道
     * @param token             经过base64编码的token，详见{@link https://help.aliyun.com/zh/live/user-guide/token-based-authentication?spm=a2c4g.11186623.0.0.315971e0tmFEgF}
     * @param channelId         房间号
     * @param userId, userName  用户id、用户名
     * @param config            joinChannel时的设置项
    '''
    @abstractmethod
    def JoinChannel(token:str, channelId:str, userId:str, userName:str, config:JoinChannelConfig) -> int:
        pass
    
    '''
     * @brief AI场景单参数加入频道，可额外设置User属性
     * @param token             经过base64编码的token，详见{@link https://help.aliyun.com/zh/live/user-guide/token-based-authentication?spm=a2c4g.11186623.0.0.315971e0tmFEgF}
     * @param userParam         与用户相关的参数，如房间号、用户id、角色等
     * @param config            joinChannel时的设置项
    '''
    @abstractmethod
    def JoinChannelWithProperty(token:str, userParam:AliEngineUserParam, config:JoinChannelConfig) -> int:
        pass

    '''
     * @brief 离开频道
    '''
    @abstractmethod
    def LeaveChannel() -> int:
        pass

    '''
     * @brief 查询是否允许推送camera track
     * @return true: 允许；false: 禁止
    '''
    @abstractmethod
    def IsLocalVideoStreamPublished() -> bool:
        pass

    '''
     * @brief 查询是否允许推送screen track
     * @return true: 允许；false: 禁止
    '''
    @abstractmethod
    def IsLocalScreenPublishEnabled() -> bool:
        pass

    '''
     * @brief 查询是否允许推送audio track
     * @return true: 允许；false: 禁止
    '''
    @abstractmethod
    def IsLocalAudioStreamPublished() -> bool:
        pass

    '''
     * @brief 查询是否允许推送simulcast (camera track)
     * @return true: 允许；false: 禁止
    '''
    @abstractmethod
    def IsDualStreamPublished() -> bool:
        pass

    '''
     * @brief 启用外部视频输入源
     * @param enable true 开启, false 关闭
     * @param type 流类型
     * @note 启用后使用PushExternalVideoFrame接口输入视频数据
    '''
    @abstractmethod
    def SetExternalVideoSource(enable:bool, sourceType:VideoSource, renderMode:RenderMode) -> int:
        pass

    '''
     * @brief 输入外部输视频, 暂不支持2k及以上的视频输入
     * @param frame 帧数据
     * @param type 流类型
     * @param 输入视频帧支持多种类型，如YUV和RGB
    '''
    @abstractmethod
    def PushExternalVideoFrame(frame:VideoDataSample, sourceType:VideoSource) -> int:
        pass

    '''
     * @brief 设置是否启用外部音频输入推流
     * @param enable true 开启，false 关闭
	 * @param sampleRate 采样率，支持的采样率 8000, 12000, 16000, 24000, 32000, 44100, 48000, 96000
	 * @param channelsPerFrame 声道数 1:单声道; 2:双声道
     * @return >=0表示成功， <0表示失败
     * @note 可通过SetExternalAudioPublishVolume设置输入音频推流音量
    '''
    @abstractmethod
    def SetExternalAudioSource(enable:bool, sampleRate:int, channelsPerFrame:int) -> int:
        pass

    '''
     * @brief 输入外部音频数据推流
     * @param audioSamples 音频数据
     * @param sampleLength 音频数据长度
     * @param timestamp 时间戳
     * @pramm delayInfo 时延信息
     * @return <0表示失败，返回值为ERR_AUDIO_BUFFER_FULL时，需要在间隔投递数据时间长度后再次重试投递
    '''
    @abstractmethod
    def PushExternalAudioFrameRawData(audioSamples:bytes, sampleLength:int, timestamp:int) -> int:
        pass

    '''
     * @brief 输入外部音频数据推流
     * @param frame 音频帧数据
     * @return <0表示失败，返回值为ERR_AUDIO_BUFFER_FULL时，需要在间隔投递数据时间长度后再次重试投递
    '''
    @abstractmethod
    def PushExternalAudioFrame(frame:AudioFrameData) -> int:
        pass

    '''
     * @brief 设置时延信息
     * @param uid 用户id
     * @param delayInfo 回答链路的时延信息
    '''
    @abstractmethod
    def SetAudioDelayInfo(self, uid: str, delayInfo: AIAudioQuestionDelay) -> int:
        pass

    '''
     * @brief 加密时延信息
     * @param delayInfo 待加密的时延信息
    '''
    @abstractmethod
    def EncryptQuestionDelayInfo(self, delayInfo: AIAudioQuestionDelay) -> int:
        pass

    '''
     * @brief 设置加密的时延信息
     * @param delayInfo 待加密的时延信息
    '''
    @abstractmethod
    def SetEncryptAudioDelayInfo(self, uid: str, sentenceId: int, delayInfo: str, extraDelay: AIExtraDelay) -> int:
        pass

    '''
     * @brief 清空音视频缓存，立即终止当前播放内容
    '''
    @abstractmethod
    def ClearDataBuffer() -> int:
        pass

    '''
     * @brief 获取剩余（未消费）缓存数据大小
     * @param dataType 缓存数据类型
     * @note 数值会通过回调接口OnGetRemainDataBufferSize返回
    '''
    @abstractmethod
    def GetRemainDataBufferSize(dataType:BufferDataType) -> int:
        pass

    '''
     * @brief 获取当前音频是否为主讲人
     * @param startTime 时间范围起始时间
     * @param endTime 时间范围结束时间
     * @note 数值会通过回调接口OnGetAudioIsMainSpeaker返回
    '''
    @abstractmethod
    def GetAudioIsMainSpeaker(startTime:int, endTime:int) -> int:
        pass

    '''
     * @brief 获取AIVad识别结果
     * @param startTime asr识别开始时间
     * @param endTime asr识别结束时间
     * @note 数值会通过回调接口OnGetAIVadInfo返回
    '''
    @abstractmethod
    def GetAIVadInfo(startTime:int, endTime:int) -> int:
        pass

    '''
     * @brief 设置外部输入音频推流混音音量
     * @param vol 音量 0-100
    '''
    @abstractmethod
    def SetExternalAudioPublishVolume(volume:int) -> int:
        pass

    '''
     * @brief 获取外部输入音频推流混音音量
     * @return vol 音量
    '''
    @abstractmethod
    def GetExternalAudioPublishVolume() -> int:
        pass

    '''
     * @brief 设置音质
    '''
    @abstractmethod
    def SetAudioProfile(audioProfile:AudioQualityMode, audioScene:AudioSceneMode) -> int:
        pass

    '''
     * @brief 向声纹降噪模块设置声纹向量
    '''
    @abstractmethod
    def SetVoiceprintVector(vector:bytes) -> int:
        pass

    '''
     * @brief 从算法模块获取自动提取的声纹向量
    '''
    @abstractmethod
    def GetVoiceprintVector() -> int:
        pass

    '''
     * @brief 发送 媒体扩展信息(SEI)， 最大长度为4*1024字节，用于业务的少量数据传输
     * @param message 扩展信息内容，可以传递4K Bytes数据
     * @param length 扩展信息长度，单位:字节
     * @param repeatCount 重复次数，用于防止网络丢包导致的消息丢失
     * @param delay 延迟多久发出去，单位:毫秒
     * @param isKeyFrame 是否只在关键帧上增加SEI
     * @return <0: 成功，-1: SDK内部错误>
    '''
    @abstractmethod
    def SendMediaExtensionMsg(message:bytes, length:int, repeatCount:int, delay:int, isKeyFrame:bool) -> int:
        pass

    '''
     * @brief 是否允许推送次要视频流
     * @param enabled 是否开启/关闭次要视频流推送
     - true: 开启次要视频流推送
     - false: 关闭次要视频流推送
     * @return
     - 0: 设置成功
     - <0: 设置失败，返回错误码
     * @note 次要视频流只在推送视频流的前提下才会推送，设置推送次要视频流时，请确认已通过 {@link PublishLocalVideoStream} 方法开启视频流推送
     * @note SDK默认设置不推送次要视频流，在加入频道前也可以调用此接口修改默认值，并在推送视频流时生效
    '''
    @abstractmethod
    def PublishLocalDualStream(enabled:bool) -> int:
        pass

    '''
     * @brief 是否推送本地视频(摄像头)流
     * @param enabled 是否开启/关闭本地视频流推送
     - true: 开启视频流推送
     - false: 关闭视频流推送
     * @return
     - 0: 设置成功
     - <0: 设置失败，返回错误码
     * @note SDK默认设置推送视频流，在加入频道前也可以调用此接口修改默认值，并在加入频道成功时生效
    '''
    @abstractmethod
    def PublishLocalVideoStream(enabled:bool) -> int:
        pass

    '''
     * @brief 是否推送本地音频流
     * @param enabled 是否开启/关闭本地音频流推送
     - true: 开启音频流推送
     - false: 关闭音频流推送
     * @return
     - 0: 设置成功
     - <0: 设置失败，返回错误码
     * @note SDK默认设置推送音频流，在加入频道前也可以调用此接口修改默认值，并在加入频道成功时生效
    '''
    @abstractmethod
    def PublishLocalAudioStream(enabled:bool) -> int:
        pass

    '''
     * @brief 开始推送屏幕流
     * @param enabled 是否开启推送屏幕流.true:开始推送，false:关闭推送 
     * @return
     * - 0: 成功
     * - <0: 失败
     * @note
    '''
    @abstractmethod
    def PublishScreenShareStream(enabled:bool) -> int:
        pass

    '''
     * @brief 停止/恢复订阅特定远端用户的音频流, 用于会中调用, 会前调用无效
     * @param uid 用户ID，从App server分配的唯一标示符
     * @param sub 是否订阅远端用户的音频流
     * @param config 远端用户音频流的参数配置
     * - true:订阅指定用户的音频流
     * - false:停止订阅指定用户的音频流
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def SubscribeRemoteAudioStream(uid:str, sub:bool, config:AudioObserverManualConfig=AudioObserverManualConfig()) -> int:
        pass

    '''
     * @brief 停止/恢复订阅远端用户的视频流, 用于会中调用, 会前调用无效
     * @param uid 用户ID，从App server分配的唯一标示符
     * @param track 视频流类型
     * - AliEngineVideoTrackNo: 无效参数，设置不会有任何效果
     * - AliEngineVideoTrackCamera: 相机流
     * - AliEngineVideoTrackScreen: 屏幕共享流
     * - AliEngineVideoTrackBoth: 相机流和屏幕共享流
     * @param sub 是否订阅远端用户的视频流
     * - true:订阅指定用户的视频流
     * - false:停止订阅指定用户的视频流
     * @return
     * - 0:设置成功
     * - <0:设置失败
     * @note
    '''
    @abstractmethod
    def SubscribeRemoteVideoStream(uid:str, videoTrack:VideoTrack, sub:bool) -> int:
        pass
        
    '''
     * @brief 设置相机流视频编码属性
     * @details 该方法用于设置相机流视频编码属性对应的视频参数，如分辨率、帧率、码率、视频方向等 所有设置的参数都有相应的范围限制，如果设置的参数不在有效范围内，SDK会自动调节
     * @param config 预定义的编码属性，详见{@link AliRTCSdk::Linux::AliEngineVideoEncoderConfiguration}
     * @note
     * - 该方法在入会前和入会后都可以调用，如果每次入会只需要设置一次相机流视频编码属性，建议在入会前调用
    '''
    @abstractmethod
    def SetVideoEncoderConfiguration(config:AliEngineVideoEncoderConfiguration) -> int:
        pass
    
    '''
     * @brief 设置屏幕流视频编码属性
     * @details 该方法用于设置屏幕流视频编码属性对应的视频参数，如分辨率、帧率、码率、视频方向等 所有设置的参数都有相应的范围限制，如果设置的参数不在有效范围内，SDK会自动调节
     * @param config 预定义的屏幕共享编码属性，详见{@link AliRTCSdk::Linux::AliEngineScreenShareEncoderConfiguration}
     * @note
     * - 该方法在入会前和入会后都可以调用，如果每次入会只需要设置一次屏幕流视频编码属性，建议在入会前调用
    '''
    @abstractmethod
    def SetScreenShareEncoderConfiguration(config:AliEngineScreenShareEncoderConfiguration) -> int:
        pass

    '''
     * @brief 设置订阅相机流格式，大流或小流
     * @param uid  userId，从App server分配的唯一标示符
     * @param streamType 流类型
     * - AliEngineVideoStreamTypeNone: 无效参数，设置不会有任何效果
     * - AliEngineVideoStreamTypeHigh: 大流
     * - AliEngineVideoStreamTypeLow: 小流
     * @return
     * - 0: 成功
     * - 非0: 失败
     * @note 推流端当前默认不推送小流，只有发送端调用了 PublishLocalDualStream(true) 打开双流模式，接收端才可以选择订阅大流还是小流，否则订阅的只能是大流；
    '''
    @abstractmethod
    def SetRemoteVideoStreamType(uid:str, streamType:AliEngineVideoStreamType) -> int:
        pass

    '''
     * @brief 开启/关闭本地摄像头采集
     * @param enable 开启/不开启
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def EnableLocalVideo(enable:bool) -> int:
        pass
    
    '''
     * @brief 停止或恢复本地视频数据发送
     * @param mute  bool类型，表明是否停止或恢复
     * @param videotrack VideoTrack类型
     * - VideoTrackNo: 没有视频通道
     * - VideoTrackCamera: 摄像头获取的视频
     * - VideoTrackScreen: 屏幕获取的视频
     * - VideoTrackBoth: 双通道获取的视频
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def MuteLocalCamera(mute:bool) -> int:
        pass

    '''
     * @brief 停止或恢复本地音频数据发送
     * @param mute  bool类型，表明是否停止或恢复外部文件推送
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def MuteLocalMic(mute:bool) -> int:
        pass

    '''
     * @brief 开启/取消音频数据回调
     * @param enabled 是否允许数据回调
     * @param audioSource 音频裸数据源类型，详见 {@link AliRtcAudioSource}
     * @param config 回调参数设置，详见{@link AliRtcAudioFrameObserverConfig}, null时默认参数为48000，1
     * @return 0: sucess
    '''
    def EnableAudioFrameObserver(enabled:bool, audioSource:AliRtcAudioSource, config:AliRtcAudioFrameObserverConfig) -> int:
        pass

    '''
     * @brief 设置用户角色
     * @param[in] clientRole 用户角色类型
     * @return 0为成功，非0失败
     * @note
     * 只可以在频道模式为AliRtcChannelProfileCommunication下调用，入会前/会议中均可设置，设置成功会收到onUpdateRoleNotify
     * @note 从Interactive转换为Live角色需要先停止推流，否则返回失败
     * @note
     * 频道模式为AliRtcChannelProfileInteractiveLive模式时，用户角色默认为AliRtcClientRoleLive
    '''
    @abstractmethod
    def SetClientRole(clientRole:AliEngineClientRole) -> int:
        pass

    '''
     * @brief 设置默认订阅视频流类型
     * @param[in] streamType 流类型，大流或小流
     * @return 0为成功，非0失败
    '''
    @abstractmethod
    def SetRemoteDefaultVideoStreamType(streamType:AliEngineVideoStreamType) -> int:
        pass

    '''
     * @brief 以观众模式入会后，若房间内的所有主播均下播，就立即回调OnError，错误码ERR_NO_PEOPLE
     * @param[in] enable 是否开启此功能，true开启，false关闭
     * @note 此功能为审核场景定制能力
    '''
    @abstractmethod
    def LeaveOnceNoStreamer(enable:bool) -> None:
        pass

    '''
     * @brief 入会时检查房间主播情况，若始终为空房间则回调OnError，错误码ERR_NO_PEOPLE
     * @brief 同时入会失败，体现为OnJoinChannelResult回调值-1
     * @param[in] seconds 检查空房间的最大时间（秒），若为0（或小于0）表示关闭入会检查功能
     * @note 此功能为审核场景定制能力，默认检查空房间的最大时长为5秒
    '''
    @abstractmethod
    def SetPeriodForCheckPeople(seconds:int) -> None:
        pass

    '''
     * @brief 设置远端视频Sample回调间隔.
     * @param[in] period 每隔period（毫秒）触发一次{@link OnRemoteVideoSample}回调，默认值为0
     * @note 此功能为审核场景定制能力，默认每帧都会通过回调抛出
    '''
    @abstractmethod
    def SetVideoCallbackPeriod(period:int) -> None:
        pass

    '''
     * @brief json格式字符串进行自定义配置，如打开关闭dataChannel
     * @param params  自定义配置信息
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def SetParameter(params:str) -> int:
        pass

    '''
     * @brief 在本地生成joinchannel所需要的token，绕过appserver完成鉴权
     * @param authInfo 自定义配置信息，在计算token前需要填充好appid、nonce、timestamp
     * @param appkey 与appid相对应，计算token所需要的key
     * @note 方法内生成单参数入会token
    '''
    @abstractmethod
    def GenerateToken(authInfo:AuthInfo, appkey:str) -> None:
        pass

    '''
     * @brief 通过dataChannel发送数据
     * @param ctrlMsg  AliRtcDataChannelMsg类型，包含待发送数据和控制指令
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def SendDataChannelMessage(ctrlMsg:AliEngineDataChannelMsg) -> int:
        pass

    '''
     * @brief 预加载音效文件
     * @param soundId 用户给该音效文件分配的ID
     * @param filePath 音效文件路径，支持本地文件和网络url
     * @return
     * - 0：成功
     * - 非0：失败
     * @note 音效相关接口为同步接口, 建议使用本地文件
    '''
    @abstractmethod
    def PreloadAudioEffect(soundId:int, filePath:str) -> int:
        pass

    '''
     * @brief 删除预加载的音效文件
     * @param soundId 用户给该音效文件分配的ID
     * @return
     * - 0：成功
     * - 非0：失败
     * @note 音效soundId应与预加载 {@link PreloadAudioEffect:} 时传入的ID相同
    '''
    @abstractmethod
    def UnloadAudioEffect(soundId:int) -> int:
        pass

    '''
     * @brief 开始播放音效
     * @details 开始播放音效接口，可以多次调用该方法传入不同的soundId和filePath，同时播放多个音效文件，音效文件播放结束后，SDK 会触发 {@link OnAudioEffectFinished:} 回调
     * @param soundId 用户给该音效文件分配的ID，每个音效均有唯一的ID，如果你已通过 {@link PreloadAudioEffect:} 将音效加载至内存，确保这里的soundId与 {@link PreloadAudioEffect:} 设置的soundId相同
     * @param filePath 文件路径，支持本地文件和网络url
     * @param config 音效播放配置，详见{@link AliRtcAudioEffectConfig}
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def PlayAudioEffect(soundId:int, filePath:str, config:AliRtcAudioEffectConfig) -> int:
        pass

    '''
     * @brief 停止播放音效
     * @param soundId 用户给该音效文件分配的ID，每个音效均有唯一的ID，如果你已通过 {@link PreloadAudioEffect:} 将音效加载至内存，确保这里的soundId与 {@link PreloadAudioEffect:} 设置的soundId相同
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def StopAudioEffect(soundId:int) -> int:
        pass

    '''
     * @brief 停止播放所有音效
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def StopAllAudioEffects() -> int:
        pass

    '''
     * @brief 设置音效推流混音音量
     * @param soundId 用户给该音效文件分配的ID
     * @param volume 推流混音音量，范围是：[0, 100]，默认值：50
     * @return
     * - 0：成功
     * - 非0：失败
     * @note 该方法需要在 {@link PlayAudioEffect:} 后调用
    '''
    @abstractmethod
    def SetAudioEffectPublishVolume(soundId:int, volume:int) -> int:
        pass

    '''
     * @brief 获取音效推流混音音量
     * @param soundId 用户给该音效文件分配的ID
     * @return
     * - [0, 100]：音效推流混音音量
     * - 其他：错误值
     * @note 音效推流混音音量有效范围为：[0, 100]，该方法需要在 {@link PlayAudioEffect:} 后调用
    '''
    @abstractmethod
    def GetAudioEffectPublishVolume(soundId:int) -> int:
        pass

    '''
     * @brief 设置音效本地播放音量
     * @param soundId 用户给该音效文件分配的ID
     * @param volume 音效本地播放音量，范围：[0, 100]，默认值：50
     * @return
     * - 0：成功
     * - 非0：失败
     * @note 该方法需要在 {@link PlayAudioEffect:} 后调用
    '''
    @abstractmethod
    def SetAudioEffectPlayoutVolume(soundId:int, volume:int) -> int:
        pass

    '''
     * @brief 获取音效本地播放音量
     * @param soundId 用户给该音效文件分配的ID
     * @return
     * - [0, 100]：音效本地播放音量
     * - 其他：错误值
     * @note 音效本地播放音量有效范围为：[0, 100]，该方法需要在 {@link PlayAudioEffect:} 后调用
    '''
    @abstractmethod
    def GetAudioEffectPlayoutVolume(soundId:int) -> int:
        pass

    '''
     * @brief 设置所有音效本地播音量
     * @param volume 音效本地播放音量，范围：[0, 100]，默认值：50
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def SetAllAudioEffectsPlayoutVolume(volume:int) -> int:
        pass

    '''
     * @brief 设置所有音效推流混音音量
     * @param volume 推流混音音量，范围是：[0, 100]，默认值：50
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def SetAllAudioEffectsPublishVolume(soundId:int) -> int:
        pass

    '''
     * @brief 暂停音效
     * @param soundId 用户给该音效文件分配的ID
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def PauseAudioEffect(soundId:int) -> int:
        pass

    '''
     * @brief 暂停所有音效
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def PauseAllAudioEffects() -> int:
        pass

    '''
     * @brief 恢复指定音效文件
     * @param soundId 用户给该音效文件分配的ID
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def ResumeAudioEffect(soundId:int) -> int:
        pass

    '''
     * @brief 恢复所有音效文件
     * @return
     * - 0：成功
     * - 非0：失败
    '''
    @abstractmethod
    def ResumeAllAudioEffects() -> int:
        pass

    '''
     * @brief 获取音频dump根目录
     * @return
     * 音频dump根目录路径
     * @note 只有开启音频dump功能，收到入会回调函数OnJoinChannelResult()后，调用该接口才能获取到路径
    '''
    @abstractmethod
    def GetAudioDumpPath() -> str:
        pass

    '''
     * @brief 获取sdk的版本号
     * @return 版本号
    '''
    @abstractmethod
    def GetSDKVersion() -> str:
        pass

    '''
     * @brief 发起语音外呼
     * @param info 外呼信息
     * @param cfg 外呼媒体参数配置
     * @param config 外呼远端音频配置
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def Dial(info:AliEngineDialInfo, cfg:AliEngineVoipConfig, config:AudioObserverManualConfig=AudioObserverManualConfig()) -> int:
        pass

    '''
     * @brief 更新外呼媒体参数配置
     * @param cfg 外呼媒体参数配置
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def DialUpdate(cfg:AliEngineVoipConfig) -> int:
        pass

    '''
     * @brief 挂断外呼
     * @return
     * - 0: 成功
     * - 非0: 失败
    '''
    @abstractmethod
    def HangUp() -> int:
        pass

    '''
     * @brief Voip模式呼入电话受理
     * @param info Sip呼入所需信息
     * @param cfg 通话的媒体配置
     * @param callbackCfg 远端音频回调配置
     * @return 
     * - 0: 接口调用成功
     * - 非0: 接口调用失败
    '''
    @abstractmethod
    def PickupIncomingCall(info:AliEngineIncomingCallInfo, cfg:AliEngineVoipConfig, callbackCfg:AudioObserverManualConfig=AudioObserverManualConfig()) -> int:
        pass

    '''
     * @brief Voip模式创建呼入媒体链接
     * @return 
     * - 0: 接口调用成功
     * - 非0: 接口调用失败
    '''
    def ConnectIncomingCall() -> int:
        pass

    '''
     * @brief Voip模式断开呼入媒体链接
     * @return 
     * - 0: 接口调用成功
     * - 非0: 接口调用失败
    '''
    def DisConnectIncomingCall() -> int:
        pass


def CreateAliRTCEngine(eventHandler:EngineEventHandlerInterface, lowPort:int, highPort:int, \
                       logPath:str, coreServicePath:str, h5mode:bool, extra:str) -> AliRTCEngineInterface:
    import AliRTCEngineImpl
    engine = AliRTCEngineImpl.AliRtcEngineImpl(eventHandler, lowPort, highPort, coreServicePath)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(engine.InitializeEngine(logPath, h5mode, extra))
    return engine
