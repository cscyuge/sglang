from AliRTCEngine import *
from AliRTCLinuxSdkDefine import *
import socket
import json
import subprocess
import resource
import random
import os
import stat
import struct
import hashlib
import base64
import asyncio
import threading
import datetime
import logging
import ctypes
import shutil
import threading
from queue import Queue

class AliRtcEngineImpl(AliRTCEngineInterface):

# --------------- macros for IPC ---------------
    __CMDSTRING = "Command"
    __REQSTRING = "Request"
    
    __CMD_BASE                                   = 0
    __CMD_CREATE_RTC_ENGINE                      = __CMD_BASE + 1
    __CMD_JOIN_CHANNEL                           = __CMD_BASE + 2
    __CMD_PUSH_VIDEO_FRAME                       = __CMD_BASE + 3
    __CMD_PUSH_AUDIO_FRAME                       = __CMD_BASE + 4
    __CMD_SET_CLIENTROLE                         = __CMD_BASE + 5
    __CMD_LEAVE_CHANNEL                          = __CMD_BASE + 6
    __CMD_SET_EXTERNAL_VIDEO_SOURCE              = __CMD_BASE + 7
    __CMD_SET_EXTERNAL_AUDIO_SOURCE              = __CMD_BASE + 8
    __CMD_PUBLISH_LOCAL_DUAL_STREAM              = __CMD_BASE + 9
    __CMD_PUBLISH_LOCAL_VIDEO_STREAM             = __CMD_BASE + 10
    __CMD_PUBLISH_LOCAL_AUDIO_STREAM             = __CMD_BASE + 11
    __CMD_PUBLISH_SCREEN_SHARE_STREAM            = __CMD_BASE + 12
    __CMD_SET_VIDEO_ENCODER_CONFIGURATION        = __CMD_BASE + 13
    __CMD_SET_SCREEN_SHARE_ENCODER_CONFIGURATION = __CMD_BASE + 14
    __CMD_SEND_MEDIA_EXTENSION_MSG               = __CMD_BASE + 15
    __CMD_SUBSCRIBE_REMOTE_AUDIO_STREAM          = __CMD_BASE + 16
    __CMD_SUBSCRIBE_REMOTE_VIDEO_STREAM          = __CMD_BASE + 17
    __CMD_SET_REMOTE_VIDEO_STREAM_TYPE           = __CMD_BASE + 18
    __CMD_MUTE_LOCAL_CAMERA                      = __CMD_BASE + 19
    __CMD_MUTE_LOCAL_MIC                         = __CMD_BASE + 20
    __CMD_SET_PARAMETER                          = __CMD_BASE + 21
    __CMD_SEND_DATA_CHANNEL_MSG                  = __CMD_BASE + 22
    __CMD_DESTROY_ENGINE                         = __CMD_BASE + 23
    __CMD_JOIN_CHANNEL_V2                        = __CMD_BASE + 24
    __CMD_SET_VIDEO_CALLBACK_PERIOD              = __CMD_BASE + 25
    __CMD_SET_AUDIO_PROFILE                      = __CMD_BASE + 26
    __CMD_SET_REMOTE_DEFAULT_VIDEO_STREAM_TYPE   = __CMD_BASE + 27
    __CMD_SET_PERIOD_FOR_CHECK_PEOPLE            = __CMD_BASE + 28
    __CMD_LEAVE_ONCE_NO_STREAMER                 = __CMD_BASE + 29
    __CMD_SET_EXTERNAL_AUDIO_PUBLISH_VOLUME      = __CMD_BASE + 30
    __CMD_CLEAR_EXTERNAL_DATA                    = __CMD_BASE + 31
    __CMD_GET_REMAIN_DATA_BUFFER_SIZE            = __CMD_BASE + 32
    __CMD_GET_AUDIO_IS_MAIN_SPEAKER              = __CMD_BASE + 33    
    __CMD_SET_VOICEPRINT_VECTOR                  = __CMD_BASE + 34
    __CMD_GET_VOICEPRINT_VECTOR                  = __CMD_BASE + 35
#    __CMD_GET_NETWORK_TIME                       = __CMD_BASE + 36
    __CMD_SET_AUDIO_DELAY_INFO                   = __CMD_BASE + 37
    __CMD_ENABLE_LOCAL_VIDEO                     = __CMD_BASE + 38
    __CMD_GET_NEED_TEST_CYCLE_ROUND              = __CMD_BASE + 39
    __CMD_GET_SDK_VERSION                        = __CMD_BASE + 40
    __CMD_JOIN_CHANNEL_V3                        = __CMD_BASE + 41
    __CMD_GET_AIVAD_INFO                         = __CMD_BASE + 42
    __CMD_DIAL                                   = __CMD_BASE + 43
    __CMD_DIAL_UPDATE                            = __CMD_BASE + 44
    __CMD_HANGUP                                 = __CMD_BASE + 45
    __CMD_ENCRYPT_DELAY_INFO                     = __CMD_BASE + 46
    __CMD_SET_AUDIO_DELAY_INFO_V2                = __CMD_BASE + 47
    __CMD_PRELOAD_AUDIO_EFFECT                   = __CMD_BASE + 48
    __CMD_UNLOAD_AUDIO_EFFECT                    = __CMD_BASE + 49
    __CMD_PLAY_AUDIO_EFFECT                      = __CMD_BASE + 50
    __CMD_STOP_AUDIO_EFFECT                      = __CMD_BASE + 51
    __CMD_STOP_ALL_AUDIO_EFFECTS                 = __CMD_BASE + 52
    __CMD_SET_AUDIO_EFFECT_PUBLISH_VOLUME        = __CMD_BASE + 53
    __CMD_GET_AUDIO_EFFECT_PUBLISH_VOLUME        = __CMD_BASE + 54
    __CMD_SET_AUDIO_EFFECT_PLAYOUT_VOLUME        = __CMD_BASE + 55
    __CMD_GET_AUDIO_EFFECT_PLAYOUT_VOLUME        = __CMD_BASE + 56
    __CMD_SET_ALL_AUDIO_EFFECTS_PLAYOUT_VOLUME   = __CMD_BASE + 57
    __CMD_SET_ALL_AUDIO_EFFECTS_PUBLISH_VOLUME   = __CMD_BASE + 58
    __CMD_PAUSE_AUDIO_EFFECT                     = __CMD_BASE + 59
    __CMD_PAUSE_ALL_AUDIO_EFFECTS                = __CMD_BASE + 60
    __CMD_RESUME_AUDIO_EFFECT                    = __CMD_BASE + 61
    __CMD_RESUME_ALL_AUDIO_EFFECTS               = __CMD_BASE + 62
    __CMD_PICKUP_INCOMING_CALL                   = __CMD_BASE + 63
    __CMD_CONNECT_INCOMING_CALL                  = __CMD_BASE + 64
    __CMD_DIS_CONNECT_INCOMING_CALL              = __CMD_BASE + 65
    __CMD_ENABLE_AUDIO_FRMAE_OBSERVER            = __CMD_BASE + 66
    
    __CALLBACK_BASE                                      = 1000
    __CALLBACK_ON_JOIN_CHANNEL                           = __CALLBACK_BASE + 1
    # CALLBACK_ON_VIDEO_FRAME                            = __CALLBACK_BASE + 2
    # CALLBACK_ON_AUDIO_FRAME                            = __CALLBACK_BASE + 3
    __CALLBACK_ON_LEAVE_CHANNEL                          = __CALLBACK_BASE + 4
    __CALLBACK_ON_REMOTE_USER_ONLINE_NOTIFY              = __CALLBACK_BASE + 5
    __CALLBACK_ON_REMOTE_USER_OFFLINE_NOTIFY             = __CALLBACK_BASE + 6
    __CALLBACK_ON_REMOTE_TRACK_AVAILABLE_NOTIFY          = __CALLBACK_BASE + 7
    __CALLBACK_ON_AUDIO_SUBSCRIBE_STATE_CHANGED          = __CALLBACK_BASE + 8
    __CALLBACK_ON_VIDEO_SUBSCRIBE_STATE_CHANGED          = __CALLBACK_BASE + 9
    __CALLBACK_ON_SUBSCRIBE_STREAM_TYPE_CHANGED          = __CALLBACK_BASE + 10
    __CALLBACK_ON_SCREEN_SHARE_SUBSCRIBE_STATE_CHANGED   = __CALLBACK_BASE + 11
    __CALLBACK_ON_SCREEN_SHARE_PUBLISH_STATE_CHANGED     = __CALLBACK_BASE + 12
    __CALLBACK_ON_DUAL_STREAM_PUBLISH_STATE_CHANGED      = __CALLBACK_BASE + 13
    __CALLBACK_ON_VIDEO_PUBLISH_STATE_CHANGED            = __CALLBACK_BASE + 14
    __CALLBACK_ON_AUDIO_PUBLISH_STATE_CHANGED            = __CALLBACK_BASE + 15
    __CALLBACK_ON_UPDATE_ROLE_NOTIFY                     = __CALLBACK_BASE + 16
    __CALLBACK_ON_ERROR                                  = __CALLBACK_BASE + 17
    __CALLBACK_ON_WARNING                                = __CALLBACK_BASE + 18
    __CALLBACK_ON_MEDIA_EXTENSION_MSG                    = __CALLBACK_BASE + 19
    __CALLBACK_ON_DATA_CHANNEL_MSG                       = __CALLBACK_BASE + 20
    __CALLBACK_ON_REMOTE_VIDEO_SAMPLE                    = __CALLBACK_BASE + 21
    __CALLBACK_ON_SUBSCRIBE_AUDIO_FRAME                  = __CALLBACK_BASE + 22
    __CALLBACK_ON_SUBSCRIBE_MIXED_AUDIO_FRAME            = __CALLBACK_BASE + 23
    __CALLBACK_ON_PUSH_AUDIO_FRAME_BUFFER_FULL           = __CALLBACK_BASE + 24
    __CALLBACK_ON_REMOTE_VIDEO_ENCODED_SAMPLE            = __CALLBACK_BASE + 25
    __CALLBACK_ON_SUBSCRIBE_AUDIO_AAC                    = __CALLBACK_BASE + 26
    __CALLBACK_ON_SUBSCRIBE_MIXED_AUDIO_AAC              = __CALLBACK_BASE + 27
    __CALLBACK_ON_PUSH_VIDEO_FRAME_BUFFER_FULL           = __CALLBACK_BASE + 28
    __CALLBACK_ON_GET_REMAIN_DATA_BUFFER_SIZE            = __CALLBACK_BASE + 29
    __CALLBACK_ON_CONNECTION_STATUS_CHANGE               = __CALLBACK_BASE + 30
    __CALLBACK_ON_GREET_READY                            = __CALLBACK_BASE + 31
    __CALLBACK_ON_AUDIO_DUMP_PATH_SET                    = __CALLBACK_BASE + 32
    __CALLBACK_ON_GET_AUDIO_IS_MAIN_SPEAKER              = __CALLBACK_BASE + 33
    __CALLBACK_ON_REMOTE_DEVICE_ID                       = __CALLBACK_BASE + 34
    __CALLBACK_ON_VOICEPRINT_VECTOR                      = __CALLBACK_BASE + 35
    __CALLBACK_ON_REMOTE_SENTENCE_STATE                  = __CALLBACK_BASE + 36
#    __CALLBACK_ON_NETWORK_TIME                           = __CALLBACK_BASE + 37
    __CALLBACK_ON_LOCAL_SENTENCE_STATE                   = __CALLBACK_BASE + 38
    __CALLBACK_ON_TEST_CYCLE_ROUND                       = __CALLBACK_BASE + 39
    __CALLBACK_ON_GET_SDK_VERSION                        = __CALLBACK_BASE + 40
    __CALLBACK_ON_GET_AIVAD_INFO                         = __CALLBACK_BASE + 41
    __CALLBACK_ON_PUSH_AUDIO_FRAME_BEGIN                 = __CALLBACK_BASE + 42
    __CALLBACK_ON_PUSH_AUDIO_FRAME_END                   = __CALLBACK_BASE + 43
    __CALLBACK_ON_PUSH_AUDIO_FRAME_BEGIN_EXT             = __CALLBACK_BASE + 44
    __CALLBACK_ON_PUSH_AUDIO_FRAME_END_EXT               = __CALLBACK_BASE + 45
    __CALLBACK_ON_REMOTE_HANGUP                          = __CALLBACK_BASE + 46
    __CALLBACK_ON_HANGUP_RESULT                          = __CALLBACK_BASE + 47
    __CALLBACK_ON_VOIP_QUALITY                           = __CALLBACK_BASE + 48
    __CALLBACK_ON_DIAL_RESULT                            = __CALLBACK_BASE + 49
    __CALLBACK_ON_DIAL_UPDATE_RESULT                     = __CALLBACK_BASE + 50
    __CALLBACK_ON_ENCRYPTION_DELAY_INFO                  = __CALLBACK_BASE + 51
    __CALLBACK_ON_DIAL_STATE_CHANGE                      = __CALLBACK_BASE + 52
    __CALLBACK_ON_AUDIO_EFFECT_FINISHED                  = __CALLBACK_BASE + 53
    __CALLBACK_ON_GET_AUDIO_EFFECT_PLAYOUT_VOLUME        = __CALLBACK_BASE + 54
    __CALLBACK_ON_GET_AUDIO_EFFECT_PUBLISH_VOLUME        = __CALLBACK_BASE + 55
    __CALLBACK_ON_PICKUP_INCOMING_CALL                   = __CALLBACK_BASE + 56
    __CALLBACK_ON_CONNECT_INCOMING_CALL_RESULT           = __CALLBACK_BASE + 57
    __CALLBACK_ON_DIS_CONNECT_INCOMING_CALL_RESULT       = __CALLBACK_BASE + 58
    __CALLBACK_ON_REMOTE_USER_SUBSCRIBE_DATA_CHANNEL     = __CALLBACK_BASE + 59
    __CALLBACK_ON_REQUEST_VIDEO_EXTERNAL_ENCODER_PARAMETER = __CALLBACK_BASE + 60
    __CALLBACK_ON_REQUEST_VIDEO_EXTERNAL_ENCODER_FRAME   = __CALLBACK_BASE + 61
    __CALLBACK_ON_STATS                                  = __CALLBACK_BASE + 62
    __CALLBACK_ON_LOCAL_VIDEO_STATS                      = __CALLBACK_BASE + 63
    __CALLBACK_ON_REMOTE_VIDEO_STATS                     = __CALLBACK_BASE + 64
    __CALLBACK_ON_LOCAL_AUDIO_STATS                      = __CALLBACK_BASE + 65
    __CALLBACK_ON_REMOTE_AUDIO_STATS                     = __CALLBACK_BASE + 66
    __CALLBACK_ON_PROFILE_STATS                          = __CALLBACK_BASE + 67
    __CALLBACK_ON_VOIP_TELEPHONE_EVENT                   = __CALLBACK_BASE + 68
    __CALLBACK_ON_PUBLISH_AUDIO_FRAME                    = __CALLBACK_BASE + 69
    __CALLBACK_ON_MIXED_ALL_AUDIO_FRAME                  = __CALLBACK_BASE + 70
    __CALLBACK_ON_FIRST_AUDIO_PACKET_RECEIVED            = __CALLBACK_BASE + 71

    __SERVER_RECV_BUF_SIZE = 65536

# --------------- private members ---------------
    __eventLoop = None
    __eventHandler = None
    __socketReader = None
    __socketWriter = None
    __socketPort = -1
    __recvTask = None
    __subProcess = None
    __stdoutThread = None
    __stderrThread = None
    __monitorTask = None
    __localCameraPublishEnabled = False
    __localScreenPublishEnabled = False
    __localAudioPublishEnabled = False
    __localDualPublishEnabled = False
    __externalAudioVolume = 0
    __joinState = False
    __leaveState = True
    __didCallRelease = False
    __lock = threading.Lock()
    __pushFirstAudio = False
    __pushFirstVideo = False
    __sendAudioStatsTs = 0
    __sendVideoStatsTs = 0
    __pushAudioFrameCnt= 0
    __pushVideoFrameCnt= 0
    __enableDumpPcm = False
    # __pcmFile = None
    __pushAudioDumpPath = ""
    __didGetNeedTestLoopbackLatency = False
    __testLoopbackLatencyEnabled = False
    __didGetAudioEffectPlayoutVolume = False
    __audioEffectPlayoutVolume = 0
    __didGetAudioEffectPublishVolume = False
    __audioEffectPublishVolume = 0
    __pushAudioSampleRate = 0
    __pushAudioChannel = 0
    __remoteAudioSampleRate = 0
    __remoteAudioChannel = 0
    __pushAudioFull = False
    __recvVideoBuffer = bytearray(3110400)
    __logger = logging.getLogger(__name__)
    __sdkVersion = ""

    async def __writeDataInternal(self, writer, data:bytes, offset:int, length:int) -> None:
        if offset < 0 or offset + length > len(data):
            return -1
        dataLength = struct.pack('!I', length)
        writer.write(dataLength + data[offset:offset+length])
        await writer.drain()


    async def __writeData(self, writer, data: bytes) -> None:
        await self.__writeDataInternal(writer, data, 0, len(data))


    async def __readDataInternal(self, reader, data:bytearray, offset:int) -> int:
        dataLength = await reader.readexactly(4)
        length = struct.unpack('!I', dataLength)[0]
        data[offset:offset+length] = await reader.readexactly(length)
        return length

    async def __readData(self, reader) -> bytes:
        dataLength = await reader.readexactly(4)
        length = struct.unpack('!I', dataLength)[0]
        data = await reader.readexactly(length)
        return data

    async def __heartbeatCoroutine(self) -> None:
        while self.__subProcess.poll() is None:
            await asyncio.sleep(1)
        else:
            self.__eventHandler.OnError(ERROR_CODE.ERR_NETWORK_DISCONNECT)

    async def __recvCoroutine(self, reader) -> None:
        message = ""
        try:
            while True:
                data = await self.__readData(reader)
                if len(data) == 0 or all(byte == 0 for byte in data):
                    continue
                message = data.decode('utf-8')
                parsed_message = json.loads(message)

                if parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_JOIN_CHANNEL:

                    # if self.__enableDumpPcm:
                    #     audio_dump_dir = self.__pushAudioDumpPath
                    #     audio_dump_file_name = audio_dump_dir + "/PushExternalAudioFrameRawData.pcm"
                    #     self.__pcmFile = open(audio_dump_file_name, 'wb')
                    #     self.__logger.info(f"[Python] pcm dump file:{audio_dump_file_name}, handler:{self.__pcmFile}")

                    code = parsed_message['joinresult']
                    channel = parsed_message['channel']
                    userid = parsed_message['userid']
                    self.__eventHandler.OnJoinChannelResult(code, channel, userid)
                    if code == 0:
                        self.__joinState = True
                        self.__leaveState = False
                    self.__logger.info(f'[Python] join channel result {code}')

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_LEAVE_CHANNEL:
                    code = parsed_message['leaveresult']
                    self.__eventHandler.OnLeaveChannelResult(code)
                    if code == 0:
                        self.__leaveState = True
                        self.__joinState = False
                    self.__logger.info(f'[Python] leave channel result {code}')

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_ERROR:
                    code = parsed_message['errorCode']
                    self.__eventHandler.OnError(ERROR_CODE(code))
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_WARNING:
                    code = parsed_message['warningCode']
                    self.__eventHandler.OnWarning(WARNNING_CODE(code))
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_USER_ONLINE_NOTIFY:
                    uid = parsed_message['uid']
                    self.__logger.info('[Python] on remote online: {uid}')
                    self.__eventHandler.OnRemoteUserOnLineNotify(uid)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_USER_OFFLINE_NOTIFY:
                    uid = parsed_message['uid']
                    self.__logger.info('[Python] on remote offline: {uid}')
                    self.__eventHandler.OnRemoteUserOffLineNotify(uid)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_TRACK_AVAILABLE_NOTIFY:
                    uid = parsed_message['uid']
                    audioTrack = AudioTrack(parsed_message['audioTrack'])
                    videoTrack = VideoTrack(parsed_message['videoTrack'])
                    self.__logger.info('[Python] on {uid}\'s remote audio track: {audioTrack.value}, remote video track {videoTrack.value}')
                    self.__eventHandler.OnRemoteTrackAvailableNotify(uid, audioTrack, videoTrack)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_VIDEO_SUBSCRIBE_STATE_CHANGED:
                    uid = parsed_message['uid']
                    oldState = AliEngineSubscribeState(parsed_message['oldState'])
                    newState = AliEngineSubscribeState(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__logger.info('[Python] on {uid}\'s v sub state changed, oldState: {oldState.value}, newState: {newState.value}')
                    self.__eventHandler.OnVideoSubscribeStateChanged(uid, oldState, newState, elapseSinceLastState, channel)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_AUDIO_SUBSCRIBE_STATE_CHANGED:
                    uid = parsed_message['uid']
                    oldState = AliEngineSubscribeState(parsed_message['oldState'])
                    newState = AliEngineSubscribeState(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__logger.info('[Python] on {uid}\'s a sub state changed, oldState: {oldState.value}, newState: {newState.value}')
                    self.__eventHandler.OnAudioSubscribeStateChanged(uid, oldState, newState, elapseSinceLastState, channel)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_SUBSCRIBE_STREAM_TYPE_CHANGED:
                    uid = parsed_message['uid']
                    oldType = AliEngineVideoStreamType(parsed_message['oldState'])
                    newType = AliEngineVideoStreamType(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__eventHandler.OnSubscribeStreamTypeChanged(uid, oldType, newType, elapseSinceLastState, channel)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_SCREEN_SHARE_SUBSCRIBE_STATE_CHANGED:
                    uid = parsed_message['uid']
                    oldState = AliEngineSubscribeState(parsed_message['oldState'])
                    newState = AliEngineSubscribeState(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__eventHandler.OnScreenShareSubscribeStateChanged(uid, oldState, newState, elapseSinceLastState, channel)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_SCREEN_SHARE_PUBLISH_STATE_CHANGED:
                    oldState = AliEnginePublishState(parsed_message['oldState'])
                    newState = AliEnginePublishState(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__eventHandler.OnScreenSharePublishStateChanged(uid, oldState, newState, elapseSinceLastState, channel)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_DUAL_STREAM_PUBLISH_STATE_CHANGED:
                    oldState = AliEnginePublishState(parsed_message['oldState'])
                    newState = AliEnginePublishState(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__eventHandler.OnDualStreamPublishStateChanged(oldState, newState, elapseSinceLastState, channel)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_VIDEO_PUBLISH_STATE_CHANGED:
                    oldState = AliEnginePublishState(parsed_message['oldState'])
                    newState = AliEnginePublishState(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__logger.info('[Python] on {uid}\'s v pub state changed, oldState: {oldState.value}, newState: {newState.value}')
                    self.__eventHandler.OnVideoPublishStateChanged(oldState, newState, elapseSinceLastState, channel)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_AUDIO_PUBLISH_STATE_CHANGED:
                    oldState = AliEnginePublishState(parsed_message['oldState'])
                    newState = AliEnginePublishState(parsed_message['newState'])
                    elapseSinceLastState = parsed_message['elapseSinceLastState']
                    channel = parsed_message['channel']
                    self.__logger.info('[Python] on {uid}\'s a pub state changed, oldState: {oldState.value}, newState: {newState.value}')
                    self.__eventHandler.OnAudioPublishStateChanged(oldState, newState, elapseSinceLastState, channel)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_UPDATE_ROLE_NOTIFY:
                    oldRole = AliEngineClientRole(parsed_message['oldRole'])
                    newRole = AliEngineClientRole(parsed_message['newRole'])
                    self.__logger.info('[Python] on update role, oldRole: {oldRole.value}, newRole: {newRole.value}')
                    self.__eventHandler.OnUpdateRoleNotify(oldRole, newRole)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_CONNECTION_STATUS_CHANGE:
                    netStatus = AliEngineConnectionStatus(parsed_message['status'])
                    netReason = AliEngineConnectionStatusChangeReason(parsed_message['reason'])
                    self.__logger.info('[Python] on connection status changed, status: {netStatus.value}. reason: {netReason.value}')
                    self.__eventHandler.OnConnectionStatusChanged(netStatus, netReason)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_MEDIA_EXTENSION_MSG:
                    uid = parsed_message['userid']
                    # message = parsed_message['message']
                    size = parsed_message['size']
                    # data = base64.b64decode(message)
                    data = await self.__readData(reader)
                    if size == len(data):
                        self.__eventHandler.OnMediaExtensionMsgReceived(uid, data, size)
                    pass

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_DATA_CHANNEL_MSG:
                    uid = parsed_message['uid']
                    networkTime = parsed_message['networkTime']
                    progress = parsed_message['progress']
                    type = AliEngineDataMsgType(parsed_message['type'])
                    dataLen = parsed_message['dataLen']
                    data = await self.__readData(reader)
                    if dataLen == len(data):
                        dataChannelMsg = AliEngineDataChannelMsg()
                        dataChannelMsg.data = data
                        dataChannelMsg.dataLen = dataLen
                        dataChannelMsg.networkTime = networkTime
                        dataChannelMsg.progress = progress
                        dataChannelMsg.type = type
                        self.__eventHandler.OnDataChannelMsg(uid, dataChannelMsg)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PUSH_AUDIO_FRAME_BUFFER_FULL:
                    isFull = parsed_message['success']
                    self.__pushAudioFull = isFull
                    self.__logger.info(f'[Python] on push audio buffer state, isfull: {isFull}')
                    self.__eventHandler.OnPushAudioFrameBufferFull(isFull)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PUSH_VIDEO_FRAME_BUFFER_FULL:
                    isFull = parsed_message['success']
                    self.__logger.info(f'[Python] on push video buffer state, isfull: {isFull}')
                    self.__eventHandler.OnPushVideoFrameBufferFull(isFull)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PUSH_AUDIO_FRAME_BEGIN:
                    frameID = parsed_message['frameID']
                    self.__logger.info(f'[Python] on push audio buffer begin: {frameID}')
                    # self.__eventHandler.OnPushAudioFrameBegin(frameID)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PUSH_AUDIO_FRAME_END:
                    frameID = parsed_message['frameID']
                    self.__logger.info(f'[Python] on push audio buffer end: {frameID}')
                    # self.__eventHandler.OnPushAudioFrameEnd(frameID)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PUSH_AUDIO_FRAME_BEGIN_EXT:
                    sentenceID = parsed_message['sentenceID']
                    sequenceID = parsed_message['sequenceID']
                    self.__logger.info(f'[Python] on push audio buffer begin, sentence:{sentenceID}, sequence:{sequenceID}')
                    self.__eventHandler.OnPushAudioFrameBegin(sentenceID, sequenceID)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PUSH_AUDIO_FRAME_END_EXT:
                    sentenceID = parsed_message['sentenceID']
                    sequenceID = parsed_message['sequenceID']
                    self.__logger.info(f'[Python] on push audio buffer end, sentence:{sentenceID}, sequence:{sequenceID}')
                    self.__eventHandler.OnPushAudioFrameEnd(sentenceID, sequenceID)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_GET_REMAIN_DATA_BUFFER_SIZE:
                    audioBufferSize = parsed_message['audioBufferSize']
                    videoBufferSize = parsed_message['videoBufferSize']
                    self.__eventHandler.OnGetRemainDataBufferSize(audioBufferSize, videoBufferSize)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_GET_AUDIO_IS_MAIN_SPEAKER:
                    status = AliEngineVoiceprintStatus()
                    status.voiceprintEnable = parsed_message['isEnable']
                    status.voiceIsMainSpeaker = parsed_message['isMainSpeaker']           
                    status.voiceIsEnrolled = parsed_message['isEnrolled']
                    status.voiceEnvironment = AliEngineVoiceEnvironment(parsed_message['environment'])
                    startTime = parsed_message['startTime']
                    endTime = parsed_message['endTime']
                    self.__eventHandler.OnGetAudioIsMainSpeaker(startTime, endTime, status)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_GET_SDK_VERSION:
                    self.__sdkVersion = parsed_message['sdkVersion']

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_GET_AIVAD_INFO:
                    status = parsed_message['status']
                    startTime = parsed_message['startTime']
                    endTime = parsed_message['endTime']
                    self.__eventHandler.OnGetAIVadInfo(startTime, endTime, status)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_ENCRYPTION_DELAY_INFO:
                    encryptDelay = parsed_message["delay"]
                    self.__eventHandler.OnEncryptQuestionDelayInfo(encryptDelay)
                    
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_VOICEPRINT_VECTOR:
                    vectorBase64 = parsed_message['vector']
                    vector = base64.b64decode(vectorBase64)
                    self.__eventHandler.OnVoiceprintVector(vector)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_TEST_CYCLE_ROUND:
                    self.__testLoopbackLatencyEnabled = parsed_message['isEnable']
                    self.__didGetNeedTestLoopbackLatency = True

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_VIDEO_SAMPLE:
                    beginTs = time.time()
                    uid = parsed_message['uid']
                    videoFrame = VideoFrame()
                    videoFrame.track = VideoTrackType(parsed_message['track'])
                    videoFrame.type = VideoFrameType(parsed_message['type'])
                    h264Frame = VideoH264Frame()
                    h264Frame.frame_ms_ = parsed_message['frame_ms']
                    h264Frame.width = parsed_message['width']
                    h264Frame.height = parsed_message['height']
                    h264Frame.frame_num_ = parsed_message['frame_num']
                    h264Frame.bufSize_ = parsed_message['frame_size']
                    h264Frame.qp = parsed_message['qp']
                    h264Frame.mv_infos = [(p[0], p[1]) for p in parsed_message['mv_infos']]
                    h264Frame.slice_type = parsed_message['slice_type']

                    offset = 0
                    length = 0
                    while offset < h264Frame.bufSize_:
                        length = await self.__readDataInternal(reader, self.__recvVideoBuffer, offset)
                        offset += length
                    h264Frame.buf_ = self.__recvVideoBuffer
                    videoFrame.frame = h264Frame
                    
                    interTs = time.time()
                    self.__eventHandler.OnRemoteVideoSample(uid, videoFrame)
                    endTs = time.time()
                    if endTs - beginTs > 0.1:
                        duration = 1000 * (endTs - beginTs)
                        interDuration = 1000 * (interTs - beginTs)
                        self.__logger.warning(f"callback video frame too long, duration: {duration}, inter duration: {interDuration}")

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_VIDEO_ENCODED_SAMPLE:
                    uid = parsed_message['uid']
                    videoFrame = VideoFrame()
                    videoFrame.track = VideoTrackType(parsed_message['track'])
                    videoFrame.type = VideoFrameType(parsed_message['type'])
                    h264Frame = VideoH264Frame()
                    h264Frame.frame_ms_ = parsed_message['frame_ms']
                    h264Frame.width = parsed_message['width']
                    h264Frame.height = parsed_message['height']
                    h264Frame.frame_num_ = parsed_message['frame_num']
                    h264Frame.bufSize_ = parsed_message['frame_size']

                    offset = 0
                    length = 0
                    while offset < h264Frame.bufSize_:
                        length = await self.__readDataInternal(reader, self.__recvVideoBuffer, offset)
                        offset += length
                    h264Frame.buf_ = self.__recvVideoBuffer
                    videoFrame.frame = h264Frame
                    self.__eventHandler.OnRemoteVideoEncodedSample(uid, videoFrame)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_SUBSCRIBE_AUDIO_FRAME:
                    beginTs = time.time()
                    uid = parsed_message['uid']
                    audioFrame = AudioFrame()
                    audioFrame.type = AudioFrameType(parsed_message['type'])
                    frame_ms = parsed_message['frame_ms']
                    sample_rates = parsed_message['sample_rates']
                    samples = parsed_message['samples']
                    pcmFrame = AudioPcmFrame(frame_ms, sample_rates, samples)
                    pcmFrame.channels_ = parsed_message['channels']
                    pcmFrame.sample_bits_ = parsed_message['sample_bits']
                    pcmFrame.streamIndex_ = parsed_message['stream_index']
                    pcmFrame.pcmBufSize_ = parsed_message['frame_size']
                    pcmFrame.pcmBuf_ = await self.__readData(reader)
                    pcmFrameSize = len(pcmFrame.pcmBuf_)
                    if pcmFrameSize == pcmFrame.pcmBufSize_:
                        audioFrame.pcm = pcmFrame
                        if self.__testLoopbackLatencyEnabled == True:
                            self.__remoteAudioSampleRate = sample_rates
                            self.__remoteAudioChannel = pcmFrame.channels_
                            self.__audio_queue.put(audioFrame.pcm.pcmBuf_)
                        self.__eventHandler.OnSubscribeAudioFrame(uid, audioFrame)
                    else:
                        self.__logger.error(f"callback audio frame fail, not enough data. Recv pcmFrame size:{pcmFrameSize}, expect:{pcmFrame.pcmBufSize_}, data:{pcmFrame.pcmBuf_}")
                        self.__eventHandler.OnError(ERROR_CODE.ERR_NETWORK_DISCONNECT)
                    endTs = time.time()
                    if endTs - beginTs > 0.1:
                        self.__logger.warning(f"callback audio frame too long, duration: {1000 * (endTs - beginTs)}")

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_SUBSCRIBE_MIXED_AUDIO_FRAME:
                    audioFrame = AudioFrame()
                    audioFrame.type = AudioFrameType(parsed_message['type'])
                    frame_ms = parsed_message['frame_ms']
                    sample_rates = parsed_message['sample_rates']
                    samples = parsed_message['samples']
                    pcmFrame = AudioPcmFrame(frame_ms, sample_rates, samples)
                    pcmFrame.channels_ = parsed_message['channels']
                    pcmFrame.sample_bits_ = parsed_message['sample_bits']
                    pcmFrame.pcmBufSize_ = parsed_message['frame_size']
                    pcmFrame.pcmBuf_ = await self.__readData(reader)
                    if len(pcmFrame.pcmBuf_) == pcmFrame.pcmBufSize_:
                        audioFrame.pcm = pcmFrame
                        self.__eventHandler.OnSubscribeMixAudioFrame(audioFrame)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_SUBSCRIBE_AUDIO_AAC:
                    uid = parsed_message['uid']
                    audioFrame = AudioFrame()
                    audioFrame.type = AudioFrameType(parsed_message['type'])
                    aacFrame = AudioAacFrame()
                    aacFrame.frame_ms_ = parsed_message['frame_ms']
                    aacFrame.sample_rates_ = parsed_message['sample_rates']
                    aacFrame.channels_ = parsed_message['channels']
                    aacFrame.aacBufSize_ = parsed_message['buf_size']
                    aacFrame.aacBuf_ = await self.__readData(reader)
                    if len(aacFrame.aacBuf_) == aacFrame.aacBufSize_:
                        audioFrame.aac = aacFrame
                        self.__eventHandler.OnSubscribeAudioAac(uid, audioFrame)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_SUBSCRIBE_MIXED_AUDIO_AAC:
                    audioFrame = AudioFrame()
                    audioFrame.type = AudioFrameType(parsed_message['type'])
                    aacFrame = AudioAacFrame()
                    aacFrame.frame_ms_ = parsed_message['frame_ms']
                    aacFrame.sample_rates_ = parsed_message['sample_rates']
                    aacFrame.channels_ = parsed_message['channels']
                    aacFrame.aacBufSize_ = parsed_message['buf_size']
                    aacFrame.aacBuf_ = await self.__readData(reader)
                    if len(aacFrame.aacBuf_) == aacFrame.aacBufSize_:
                        audioFrame.aac = aacFrame
                        self.__eventHandler.OnSubscribeMixedAudioAac(audioFrame)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_MIXED_ALL_AUDIO_FRAME:
                    audioFrame = AudioFrame()
                    audioFrame.type = AudioFrameType(parsed_message['type'])
                    frame_ms = parsed_message['frame_ms']
                    sample_rates = parsed_message['sample_rates']
                    samples = parsed_message['samples']
                    pcmFrame = AudioPcmFrame(frame_ms, sample_rates, samples)
                    pcmFrame.channels_ = parsed_message['channels']
                    pcmFrame.sample_bits_ = parsed_message['sample_bits']
                    pcmFrame.pcmBufSize_ = parsed_message['frame_size']
                    pcmFrame.pcmBuf_ = await self.__readData(reader)
                    if len(pcmFrame.pcmBuf_) == pcmFrame.pcmBufSize_:
                        audioFrame.pcm = pcmFrame
                        self.__eventHandler.OnMixedAllAudioFrame(audioFrame)
                        
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PUBLISH_AUDIO_FRAME:
                    audioFrame = AudioFrame()
                    audioFrame.type = AudioFrameType(parsed_message['type'])
                    frame_ms = parsed_message['frame_ms']
                    sample_rates = parsed_message['sample_rates']
                    samples = parsed_message['samples']
                    pcmFrame = AudioPcmFrame(frame_ms, sample_rates, samples)
                    pcmFrame.channels_ = parsed_message['channels']
                    pcmFrame.sample_bits_ = parsed_message['sample_bits']
                    pcmFrame.pcmBufSize_ = parsed_message['frame_size']
                    pcmFrame.pcmBuf_ = await self.__readData(reader)
                    if len(pcmFrame.pcmBuf_) == pcmFrame.pcmBufSize_:
                        audioFrame.pcm = pcmFrame
                        self.__eventHandler.OnPublishAudioFrame(audioFrame)
                        
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_GREET_READY:
                    isReady = parsed_message['ready']
                    self.__eventHandler.OnGreetReady(isReady)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_DEVICE_ID:
                    uid = parsed_message['uid']
                    udid = parsed_message['udid']
                    self.__eventHandler.OnRemoteDeviceId(uid, udid)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_SENTENCE_STATE:
                    uid = parsed_message['uid']
                    isEnd = parsed_message['end']
                    delayInfo = AIAudioQuestionDelay()
                    if 'audio_question_delay' in parsed_message:
                        audioQuestionDelay = parsed_message['audio_question_delay']
                        delayInfo.ai_agent_audio_asr_cost = audioQuestionDelay['ai_agent_audio_asr_cost']
                        delayInfo.ai_question_capture_timestamp = audioQuestionDelay['ai_question_capture_timestamp']
                        delayInfo.ai_answer_capture_timestamp = audioQuestionDelay['ai_answer_capture_timestamp']
                        delayInfo.ai_sentence_id = audioQuestionDelay['ai_sentence_id']
                        delayInfo.ai_client_audio_pub_total_delay = audioQuestionDelay["ai_client_audio_total_pub_delay"]
                        delayInfo.ai_client_audio_source_cost = audioQuestionDelay['ai_client_audio_source_cost']
                        delayInfo.ai_client_audio_mixer_cost = audioQuestionDelay['ai_client_audio_mixer_cost']
                        delayInfo.ai_client_audio_encoder_cost = audioQuestionDelay['ai_client_audio_encoder_cost']
                        delayInfo.ai_client_audio_netsdk_thr_cost = audioQuestionDelay['ai_client_audio_netsdk_thr_cost']
                        delayInfo.ai_client_audio_qos_thr_cost = audioQuestionDelay['ai_client_audio_qos_thr_cost']
                        delayInfo.ai_client_audio_pacer_cost = audioQuestionDelay['ai_client_audio_pacer_cost']
                        delayInfo.ai_client_up_half_rtt = audioQuestionDelay['ai_client_up_half_rtt']
                        delayInfo.ai_linux_down_half_rtt = audioQuestionDelay['ai_linux_down_half_rtt']
                        if 'ai_linux_grtn_node_delay' in audioQuestionDelay:
                            for node in audioQuestionDelay['ai_linux_grtn_node_delay']:
                                grtn_node_delay = GrtnNodeDelay()
                                grtn_node_delay.grtn_node_internal_rtt_half = node['grtn_node_internal_rtt_half']
                                grtn_node_delay.grtn_node_bef_pacer = node['grtn_node_bef_pacer']
                                grtn_node_delay.grtn_node_pacer_cost = node['grtn_node_pacer_cost']
                                delayInfo.ai_linux_grtn_node_delay.append(grtn_node_delay)
                        delayInfo.ai_linux_audio_sub_total_delay = audioQuestionDelay['ai_linux_audio_total_sub_delay']
                        delayInfo.ai_linux_audio_receive_cost = audioQuestionDelay['ai_linux_audio_receive_cost']
                        delayInfo.ai_linux_audio_neteq_cost = audioQuestionDelay['ai_linux_audio_neteq_cost']
                        delayInfo.ai_linux_audio_remote_source_cost = audioQuestionDelay['ai_linux_audio_remote_source_cost']
                        delayInfo.ai_linux_audio_play_mixer_cost = audioQuestionDelay['ai_linux_audio_play_mixer_cost']
                        delayInfo.ai_linux_audio_player_cost = audioQuestionDelay['ai_linux_audio_player_cost']
                        delayInfo.ai_agent_audio_asr_cost = audioQuestionDelay['ai_agent_audio_asr_cost']
                        delayInfo.ai_agent_audio_llm_cost = audioQuestionDelay['ai_agent_audio_llm_cost']
                        delayInfo.ai_agent_audio_tts_cost = audioQuestionDelay['ai_agent_audio_tts_cost']
                        # delayInfo.ai_question_agent_begin_timestamp = audioQuestionDelay['ai_question_agent_begin_timestamp']
                        delayInfo.ai_agent_audio_total_cost = audioQuestionDelay['ai_agent_audio_total_cost']
                        delayInfo.ai_linux_audio_process_smart_denoise_delay = audioQuestionDelay['ai_linux_audio_process_smart_denoise_delay']
                    self.__eventHandler.OnRemoteSentenceState(uid, isEnd, delayInfo)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_LOCAL_SENTENCE_STATE:
                    uid = parsed_message['uid']
                    isBegin = parsed_message['begin']
                    delayInfo = AIAudioQuestionDelay()
                    delayInfo.ai_agent_audio_asr_cost = parsed_message['asr']
                    delayInfo.ai_agent_audio_llm_cost = parsed_message['llm']
                    delayInfo.ai_agent_audio_tts_cost = parsed_message['tts']
                    # delayInfo.ai_agent_audio_total_cost = parsed_message['total']
                    delayInfo.ai_sentence_id = parsed_message['sentence']
                    self.__eventHandler.OnLocalSentenceState(uid, isBegin, delayInfo)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_AUDIO_DUMP_PATH_SET:
                    self.__pushAudioDumpPath = parsed_message['audiopath']

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_AUDIO_EFFECT_FINISHED:
                    soundId = parsed_message['soundId']
                    self.__eventHandler.OnAudioEffectFinished(soundId)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_GET_AUDIO_EFFECT_PLAYOUT_VOLUME:
                    __audioEffectPlayoutVolume = parsed_message['volume']
                    self.__didGetAudioEffectPlayoutVolume = True

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_GET_AUDIO_EFFECT_PUBLISH_VOLUME:
                    __audioEffectPublishVolume = parsed_message['volume']
                    self.__didGetAudioEffectPublishVolume = True

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_HANGUP:
                    self.__eventHandler.OnRemoteHangUp()

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_HANGUP_RESULT:
                    code = parsed_message['code']
                    self.__eventHandler.OnHangUpResult(code)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_VOIP_QUALITY:
                    code = parsed_message['code']
                    self.__eventHandler.OnVoipQuality(code)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_DIAL_RESULT:
                    code = parsed_message['code']
                    self.__eventHandler.OnDialResult(code)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_DIAL_UPDATE_RESULT:
                    code = parsed_message['code']
                    self.__eventHandler.OnDialUpdateResult(code)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_DIAL_STATE_CHANGE:
                    code = parsed_message['code']
                    self.__eventHandler.OnDialStateChange(code)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PICKUP_INCOMING_CALL:
                    incomingCallInfo = AliEnginePickupIncomingCallInfo()
                    incomingCallInfo.sdpInfo = parsed_message['sdpInfo']
                    incomingCallInfo.result = parsed_message['result']
                    self.__eventHandler.OnPickupIncomingCall(incomingCallInfo)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_CONNECT_INCOMING_CALL_RESULT:
                    code = parsed_message['code']
                    self.__eventHandler.OnConnectIncomingCallResult(code)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_DIS_CONNECT_INCOMING_CALL_RESULT:
                    code = parsed_message['code']
                    self.__eventHandler.OnDisConnectIncomingCallResult(code)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_VOIP_TELEPHONE_EVENT:
                    eventStr = parsed_message['eventStr']
                    eventCode = parsed_message['eventCode']
                    endFlag = parsed_message['endFlag']
                    volume = parsed_message['volume']
                    timestampIncrement = parsed_message['timestampIncrement']
                    duration = parsed_message['duration']
                    self.__eventHandler.OnVoipTelephoneEvent(eventStr, eventCode, endFlag, volume, timestampIncrement, duration)
                    
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_USER_SUBSCRIBE_DATA_CHANNEL:
                    uid = parsed_message['uid']
                    self.__eventHandler.OnRemoteUserSubscribedDataChannel(uid)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REQUEST_VIDEO_EXTERNAL_ENCODER_PARAMETER:
                    videoTrack = VideoTrack(parsed_message['videoTrack'])
                    parameter = AliEngineVideoExternalEncoderParameter()
                    parameter.width = parsed_message['width']
                    parameter.height = parsed_message['height']
                    parameter.frame_rate = parsed_message['frameRate']
                    parameter.bitrate_bps = parsed_message['bitrateBps']
                    self.__eventHandler.OnRequestVideoExternalEncoderParameter(videoTrack, parameter)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REQUEST_VIDEO_EXTERNAL_ENCODER_FRAME:
                    videoTrack = VideoTrack(parsed_message['videoTrack'])
                    frameType = AliEngineVideoEncodedFrameType(parsed_message['frameType'])
                    dropFrame = parsed_message['dropFrame']
                    self.__eventHandler.OnRequestVideoExternalEncoderFrame(videoTrack, frameType, dropFrame)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_STATS:
                    statsInfo = AliEngineStats()
                    statsInfo.availableSendBitrate = parsed_message['availableSendBitrate']
                    statsInfo.sentKBitrate = parsed_message['sentKBitrate']
                    statsInfo.rcvdKBitrate = parsed_message['rcvdKBitrate']
                    statsInfo.sentBytes = parsed_message['sentBytes']
                    statsInfo.rcvdBytes = parsed_message['rcvdBytes']
                    statsInfo.videoRcvdKBitrate = parsed_message['videoRcvdKBitrate']
                    statsInfo.videoSentKBitrate = parsed_message['videoSentKBitrate']
                    statsInfo.systemCpu = parsed_message['systemCpu']
                    statsInfo.appCpu = parsed_message['appCpu']
                    statsInfo.callDuration = parsed_message['callDuration']
                    statsInfo.sentLossRate = parsed_message['sentLossRate']
                    statsInfo.sentLossPkts = parsed_message['sentLossPkts']
                    statsInfo.sentExpectedPkts = parsed_message['sentExpectedPkts']
                    statsInfo.rcvdLossRate = parsed_message['rcvdLossRate']
                    statsInfo.rcvdLossPkts = parsed_message['rcvdLossPkts']
                    statsInfo.rcvdExpectedPkts = parsed_message['rcvdExpectedPkts']
                    statsInfo.lastmileDelay = parsed_message['lastmileDelay']
                    self.__eventHandler.OnStats(statsInfo)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_LOCAL_VIDEO_STATS:
                    statsInfo = AliEngineLocalVideoStats()
                    statsInfo.track = VideoTrack(parsed_message['track'])
                    statsInfo.targetEncodeBitrate = parsed_message['targetEncodeBitrate']
                    statsInfo.actualEncodeBitrate = parsed_message['actualEncodeBitrate']
                    statsInfo.sentBitrate = parsed_message['sentBitrate']
                    statsInfo.sentBitrateKbps = parsed_message['sentBitrateKbps']
                    statsInfo.sentFps = parsed_message['sentFps']
                    statsInfo.encodeFps = parsed_message['encodeFps']
                    statsInfo.captureFps = parsed_message['captureFps']
                    statsInfo.avgQpPerSec = parsed_message['avgQpPerSec']
                    statsInfo.rtt = parsed_message['rtt']
                    statsInfo.sendBytes = parsed_message['sendBytes']
                    statsInfo.renderFps = parsed_message['renderFps']
                    self.__eventHandler.OnLocalVideoStats(statsInfo)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_VIDEO_STATS:
                    statsInfo = AliEngineRemoteVideoStats()
                    statsInfo.track = VideoTrack(parsed_message['track'])
                    statsInfo.userId = parsed_message['userId']
                    statsInfo.width = parsed_message['width']
                    statsInfo.height = parsed_message['height']
                    statsInfo.decodeFps = parsed_message['decodeFps']
                    statsInfo.renderFps = parsed_message['renderFps']
                    statsInfo.frozenTimes = parsed_message['frozenTimes']
                    statsInfo.videoTotalFrozenTime = parsed_message['videoTotalFrozenTime']
                    statsInfo.videoTotalFrozenRate = parsed_message['videoTotalFrozenRate']
                    statsInfo.rtpCount = parsed_message['rtpCount']
                    statsInfo.rtpLoss = parsed_message['rtpLoss']
                    statsInfo.rtt = parsed_message['rtt']
                    statsInfo.recvBitrate = parsed_message['recvBitrate']
                    statsInfo.rcvdBitrateKbps = parsed_message['rcvdBitrateKbps']
                    statsInfo.e2eDelay = parsed_message['e2eDelay']
                    statsInfo.frameLoss = parsed_message['frameLoss']
                    statsInfo.pullBytes = parsed_message['pullBytes']
                    statsInfo.packetLossRate = parsed_message['packetLossRate']
                    self.__eventHandler.OnRemoteVideoStats(statsInfo)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_LOCAL_AUDIO_STATS:
                    statsInfo = AliEngineLocalAudioStats()
                    statsInfo.track = AudioTrack(parsed_message['track'])
                    statsInfo.scene = AudioSceneMode(parsed_message['scene'])
                    statsInfo.sentBitrate = parsed_message['sentBitrate']
                    statsInfo.sentBitrateKbps = parsed_message['sentBitrateKbps']
                    statsInfo.sentSamplerate = parsed_message['sentSamplerate']
                    statsInfo.numChannel = parsed_message['numChannel']
                    statsInfo.inputLevel = parsed_message['inputLevel']
                    statsInfo.rtt = parsed_message['rtt']
                    statsInfo.sendBytes = parsed_message['sendBytes']
                    self.__eventHandler.OnLocalAudioStats(statsInfo)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_REMOTE_AUDIO_STATS:
                    statsInfo = AliEngineRemoteAudioStats()
                    statsInfo.track = AudioTrack(parsed_message['track'])
                    statsInfo.userId = parsed_message['userId']
                    statsInfo.quality = parsed_message['quality']
                    statsInfo.networkTransportDelay = parsed_message['networkTransportDelay']
                    statsInfo.jitterBufferDelay = parsed_message['jitterBufferDelay']
                    statsInfo.audioLossRate = parsed_message['audioLossRate']
                    statsInfo.rcvdBitrate = parsed_message['rcvdBitrate']
                    statsInfo.rcvdBitrateKbps = parsed_message['rcvdBitrateKbps']
                    statsInfo.totalFrozenTimes = parsed_message['totalFrozenTimes']
                    statsInfo.audioTotalFrozenTime = parsed_message['audioTotalFrozenTime']
                    statsInfo.audioTotalFrozenRate = parsed_message['audioTotalFrozenRate']
                    statsInfo.rtt = parsed_message['rtt']
                    statsInfo.e2eDelay = parsed_message['e2eDelay']
                    statsInfo.ai_e2eDelay = parsed_message['ai_e2eDelay']
                    statsInfo.pullBytes = parsed_message['pullBytes']
                    self.__eventHandler.OnRemoteAudioStats(statsInfo)

                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_PROFILE_STATS:
                    typeFlag = parsed_message['type']
                    profileStats = parsed_message['profileStats']
                    self.__eventHandler.OnProfileStats(typeFlag, profileStats)
                
                elif parsed_message[self.__REQSTRING] == self.__CALLBACK_ON_FIRST_AUDIO_PACKET_RECEIVED:
                    uid = parsed_message['uid']
                    timeCost = parsed_message['timeCost']
                    self.__eventHandler.OnFirstAudioPacketReceived(uid, timeCost)
                    
                elif parsed_message[self.__REQSTRING] != None:
                    # print(f"[Python] Unknown command: {parsed_message[self.__REQSTRING]}")
                    pass

        except json.JSONDecodeError:
            print(f"invalid json: {message}")
        except asyncio.CancelledError:
            if self.__didCallRelease != True:
                print("Recv coroutine has been cancelled")
        except Exception as e:
            print(f"Exception occurred: {e}")


    def SetCoreUnlimited(self) -> None:
        resource.setrlimit(resource.RLIMIT_CORE,(resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        self.__logger.info(f"[Python] DidSetCoreUnlimited")

    def ReadStdOut(self):
        if self.__subProcess and self.__subProcess.stdout:
            for line in iter(self.__subProcess.stdout.readline, ''):
                if self.__didCallRelease or self.__subProcess.poll() is not None:
                    break
                # print(line, end='', flush=True)
                outputStr = line.decode().strip()
                print(outputStr)
                self.__logger.info('[Cpp]'+ outputStr)
                if outputStr.startswith("Bind port: "):
                    self.__socketPort = outputStr.split()[-1]
            if self.__subProcess.stdout:
                self.__subProcess.stdout.close()

    def ReadStdErr(self):
        if self.__subProcess and self.__subProcess.stderr:
            for line in iter(self.__subProcess.stderr.readline, ''):
                if self.__didCallRelease or self.__subProcess.poll() is not None:
                    break
                # print(line, end='', flush=True)
                outputStr = line.decode().strip()
                print(outputStr)
                self.__logger.info('[Cpp]err '+ outputStr)
            if self.__subProcess.stderr:
                self.__subProcess.stderr.close()

    def __init__(self, eventHandler:EngineEventHandlerInterface, lowPort:int, highPort:int, \
               coreServicePath:str) -> None:
        if eventHandler == None or lowPort > highPort:
            raise ValueError("[Python] Parameter error, cannot create RTC instance")
        self._artc_loop = None  # Set by CreateAliRTCEngine after init
        self._artc_thread = None
        self.__audio_queue = Queue()
        self.__push_thread = None
        self.__thread_running = False
        self.__eventHandler = eventHandler
        serverPort = 0
        portTryCnt = 0
        while serverPort == 0:
            portTest = random.randint(lowPort, highPort)
            portTryCnt = portTryCnt+1
            if(portTryCnt >= 1000):
                break
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.bind(('localhost', portTest))
                    serverPort = portTest
                    break
                except OSError:
                    continue        

        if serverPort > 0 and isinstance(coreServicePath, str) and len(coreServicePath) > 0:
            self.__logger.info(f"[Python] core dump status:")
            self.__logger.info(resource.getrlimit(resource.RLIMIT_CORE))
            st = os.stat(coreServicePath)
            os.chmod(coreServicePath, st.st_mode | stat.S_IEXEC)
            parent_directory = os.path.dirname(coreServicePath)
            ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
            if ld_library_path and ld_library_path.find(parent_directory) == -1:
                ld_library_path = f"{parent_directory}:{ld_library_path}"
            else:
                ld_library_path = parent_directory
            os.environ['LD_LIBRARY_PATH'] = ld_library_path
            print(f"serverPort is {serverPort}")
            self.__subProcess = subprocess.Popen([coreServicePath, str(serverPort)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=self.SetCoreUnlimited)
            self.__stdoutThread = threading.Thread(target=self.ReadStdOut)
            self.__stdoutThread.daemon = True
            self.__stdoutThread.start()
            self.__stderrThread = threading.Thread(target=self.ReadStdErr)
            self.__stderrThread.daemon = True
            self.__stderrThread.start()
            while True:
                if self.__socketPort != -1:
                    break
        else:
            raise ValueError("[Python] Parameter error, cannot create RTC instance")



    async def InitializeEngine(self, logPath:str, h5mode:bool, extra:str) -> None:
        max_retries = 20
        retry_delay = 0.05  # 50ms

        needInitLogger = True
        if needInitLogger:
            self.__logger.setLevel(logging.INFO)
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S%f")[:-3]
            loggingDir = logPath + '/Ali_RTC_Log_Biz/'
            if not os.path.exists(loggingDir):
                try:
                    os.makedirs(loggingDir)
                except FileExistsError:
                    print(f"[Python] File exists: {loggingDir}")
            loggingFileName = 'Python-' + formatted_time + '.log'
            loggingPath = os.path.join(loggingDir, loggingFileName)
            file_handler = logging.FileHandler(loggingPath)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s.%(msecs)03d - %(thread)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.__logger.addHandler(file_handler)
        else:
            loggingDir = logPath + '/Ali_RTC_Log_Biz/'
            if os.path.exists(loggingDir):
                try:
                    shutil.rmtree(loggingDir)
                except OSError as error:
                    print(f"Error: {error.strerror}")
        for attempt in range(max_retries):
            try:
                self.__socketReader, self.__socketWriter = await \
                    asyncio.open_connection('localhost', str(self.__socketPort))
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise Exception("Failed to connect to the server after 20 attempts.") from e

        __logPath = logPath
        if not isinstance(logPath, str) or len(logPath) == 0:
            __logPath = "/tmp"

        self.__recvTask = asyncio.ensure_future(self.__recvCoroutine(self.__socketReader))
        self.__monitorTask = asyncio.ensure_future(self.__heartbeatCoroutine())

        # Python层增加PCM DUMP
        parsed_json = json.loads(extra)
        if "user_specified_audio_dump" in parsed_json:
            self.__enableDumpPcm = parsed_json["user_specified_audio_dump"]
        self.__logger.info(f"[Python] dump pcm:{self.__enableDumpPcm}")

        jobj = {
            self.__CMDSTRING: self.__CMD_CREATE_RTC_ENGINE,
            "logPath": __logPath,
            "h5mode" : 1 if h5mode else 0,
            "extra" : extra
        }
        jobj_str = json.dumps(jobj)
        await self.__writeData(self.__socketWriter, jobj_str.encode('utf-8'))
        self.__logger.info("[Python] initialize engine!")

    async def __UninitializeEngine(self, writer) -> None:
        self.__logger.info("[Python] uninitialize engine!")
        if self.__thread_running:
            self.__thread_running = False
        if self.__push_thread is not None:
            self.__push_thread.join()
            self.__push_thread = None
        if self.__recvTask != None:
            self.__recvTask.cancel()
            self.__recvTask = None
        if self.__monitorTask != None:
            self.__monitorTask.cancel()
            self.__monitorTask = None
        if writer != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_DESTROY_ENGINE
            }
            jobj_str = json.dumps(jobj)
            await self.__writeData(writer, jobj_str.encode('utf-8'))
            exit_code = self.__subProcess.wait()
            self.__logger.info(f"[Python] sub Process did close with exit code:{exit_code}")
            if self.__stdoutThread:
                self.__stdoutThread.join(timeout=1)
            if self.__stderrThread:
                self.__stderrThread.join(timeout=1)
            writer.close()
            if hasattr(writer, 'wait_closed'):
                await writer.wait_closed()
        self.__eventHandler = None
        self.__pushFirstAudio = False
        self.__pushFirstVideo = False
        self.__pushVideoFrameCnt = 0
        self.__pushAudioFrameCnt = 0
        self.__logger.info("[Python] finish uninitialize engine!")

    def Release(self) -> int:
        with self.__lock:
            async def internalSleep(seconds: float) -> None:
                await asyncio.sleep(seconds)
            loop = self._artc_loop
            init = time.time()
            while self.__joinState == True and self.__leaveState == False:
                asyncio.run_coroutine_threadsafe(internalSleep(0.5), loop).result()
                if time.time() - init >= 10:
                    self.__logger.error("Unable to destroy, please leave channel first")
                    print(f"[Python] Unable to destroy, please leave channel first")
                    return -1
            self.__didCallRelease = True
            asyncio.run_coroutine_threadsafe(self.__UninitializeEngine(self.__socketWriter), loop).result()       
            # loop.close()
            return 0            


    def GetEventHandler(self) -> EngineEventHandlerInterface:
        return self.__eventHandler

    
    def JoinChannelFromServer(self, authInfo: AuthInfo, config: JoinChannelConfig) -> int:
        self.__logger.info("[Python] join channel from server begin...")
        if not isinstance(authInfo, AuthInfo) or not isinstance(config, JoinChannelConfig):
            self.__eventHandler.OnError(ERROR_CODE.ERR_JOIN_CONFIG_INVALID)
            return -1
        if not isinstance(authInfo.appid, str) or not isinstance(authInfo.channel, str) or \
            not isinstance(authInfo.userid, str):
            self.__eventHandler.OnError(ERROR_CODE.ERR_JOIN_CONFIG_INVALID)
            return -1

        if self.__socketWriter != None:
            jobjAuthInfo = {
            "channel": authInfo.channel,
            "userid": authInfo.userid,
            "username": authInfo.username,
            "appid": authInfo.appid,
            "nonce": authInfo.nonce if authInfo.nonce is not None else "",
            "role": authInfo.role if authInfo.role is not None else "",
            "token": authInfo.token,
            "timestamp": authInfo.timestamp
            }

            jobjConfig = {
                "isAudioOnly": config.isAudioOnly,
                "channelProfile": config.channelProfile.value,
                "publishMode": config.publishMode.value,
                "subscribeMode": config.subscribeMode.value,
                "publishAvsyncMode": config.publishAvsyncMode.value,
                "publishAvsyncWithPtsMaxAudioCacheSize": config.publishAvsyncWithPtsMaxAudioCacheSize,
                "publishAvsyncWithPtsMaxVideoCacheSize": config.publishAvsyncWithPtsMaxVideoCacheSize,
                "subscribeVideoFormat": config.subscribeVideoFormat.value,
                "subscribeAudioFormat": config.subscribeAudioFormat.value,
                "ttsVad": config.enableTtsCallback
            }
            jobj = {
                self.__CMDSTRING: self.__CMD_JOIN_CHANNEL,
                "authInfo": jobjAuthInfo,
                "config": jobjConfig
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1
        
    def JoinChannel(self, token:str, channelId:str, userId:str, userName:str, config:JoinChannelConfig) -> int:
        self.__logger.info("[Python] join channel begin...")
        if not isinstance(token, str) or not isinstance(channelId, str) or \
            not isinstance(userId, str) or not isinstance(config, JoinChannelConfig):
            self.__eventHandler.OnError(ERROR_CODE.ERR_JOIN_CONFIG_INVALID)
            return -1

        if self.__socketWriter != None:
            jobjConfig = {
                "isAudioOnly": config.isAudioOnly,
                "channelProfile": config.channelProfile.value,
                "publishMode": config.publishMode.value,
                "subscribeMode": config.subscribeMode.value,
                "publishAvsyncMode": config.publishAvsyncMode.value,
                "publishAvsyncWithPtsMaxAudioCacheSize": config.publishAvsyncWithPtsMaxAudioCacheSize,
                "publishAvsyncWithPtsMaxVideoCacheSize": config.publishAvsyncWithPtsMaxVideoCacheSize,
                "subscribeVideoFormat": config.subscribeVideoFormat.value,
                "subscribeAudioFormat": config.subscribeAudioFormat.value,
                "ttsVad": config.enableTtsCallback
            }
            jobj = {
                self.__CMDSTRING: self.__CMD_JOIN_CHANNEL_V2,
                "token": token,
                "channel": channelId,
                "userid": userId,
                "username": userName,
                "config": jobjConfig
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1


    def JoinChannelWithProperty(self, token:str, userParam:AliEngineUserParam, config:JoinChannelConfig) -> int:
        self.__logger.info("[Python] join channel with property begin...")
        if not isinstance(token, str) or not isinstance(userParam, AliEngineUserParam) or \
            not isinstance(config, JoinChannelConfig):
            self.__eventHandler.OnError(ERROR_CODE.ERR_JOIN_CONFIG_INVALID)
            return -1

        if self.__socketWriter != None:
            jobjConfig = {
                "isAudioOnly": config.isAudioOnly,
                "channelProfile": config.channelProfile.value,
                "publishMode": config.publishMode.value,
                "subscribeMode": config.subscribeMode.value,
                "publishAvsyncMode": config.publishAvsyncMode.value,
                "publishAvsyncWithPtsMaxAudioCacheSize": config.publishAvsyncWithPtsMaxAudioCacheSize,
                "publishAvsyncWithPtsMaxVideoCacheSize": config.publishAvsyncWithPtsMaxVideoCacheSize,
                "subscribeVideoFormat": config.subscribeVideoFormat.value,
                "subscribeAudioFormat": config.subscribeAudioFormat.value,
                "ttsVad": config.enableTtsCallback
            }
            jobj = {
                self.__CMDSTRING: self.__CMD_JOIN_CHANNEL_V3,
                "token": token,
                "channel": userParam.channelId,
                "userid": userParam.userId,
                "username": userParam.userName,
                "capabilityProfile": userParam.capabilityProfile.value,
                "useVoip": userParam.useVoip,
                "config": jobjConfig
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1


    def LeaveChannel(self) -> int:
        self.__logger.info("[Python] leave channel begin...")
        # if self.__pcmFile:
        #     self.__pcmFile.close()
        #     self.__pcmFile = None
        #     self.__logger.info("[Python] pcm file closed")

        if self.__socketWriter != None:
            with self.__lock:
                if self.__thread_running:
                    self.__thread_running = False
                    if self.__push_thread is not None:
                        self.__push_thread.join()
                        self.__push_thread = None
            jobj = {
                self.__CMDSTRING: self.__CMD_LEAVE_CHANNEL,
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1


    def IsLocalVideoStreamPublished(self) -> bool:
        return self.__localCameraPublishEnabled


    def IsLocalScreenPublishEnabled(self) -> bool:
        return self.__localScreenPublishEnabled

    
    def IsLocalAudioStreamPublished(self) -> bool:
        return self.__localAudioPublishEnabled

    
    def IsDualStreamPublished(self) -> bool:
        return self.__localDualPublishEnabled

    
    def SetExternalVideoSource(self, enable:bool, sourceType:VideoSource, renderMode:RenderMode) -> int:
        self.__logger.info(f"[Python] set external video source {enable}")
        if not isinstance(enable, bool) or not isinstance(sourceType, VideoSource) or not isinstance(renderMode, RenderMode):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_EXTERNAL_VIDEO_SOURCE,
                "enable": enable,
                "source": sourceType.value,
                "renderMode": renderMode.value
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    
    def PushExternalVideoFrame(self, frame:VideoDataSample, sourceType:VideoSource) -> int:
        beginTs = time.time()
        if not isinstance(frame, VideoDataSample) or not isinstance(sourceType, VideoSource):
            return -1
        if len(frame.data) < frame.dataLen:
            self.__logger.error('[Python] not enough video data')
            return -1
        if len(frame.data) == 0 or frame.dataLen <= 0:
            return -1
        if self.__socketWriter != None:
            if not self.__pushFirstVideo:
                self.__logger.info("[Python] push first video frame...")
                self.__pushFirstVideo = True
            # lock_address = ctypes.addressof(ctypes.py_object(self.__lock))
            # print(f'Lock address for video: {lock_address:#x}')
            jobj = {
                self.__CMDSTRING: self.__CMD_PUSH_VIDEO_FRAME,
                "source": sourceType.value,
                "format": frame.format.value,
                "bufferType": frame.bufferType.value,
                "strideY": frame.strideY,
                "strideU": frame.strideU,
                "strideV": frame.strideV,
                "height": frame.height,
                "width": frame.width,
                "rotation": frame.rotation,
                "timestamp": frame.timeStamp,
                "dataLen": frame.dataLen
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
                interTs = time.time()

                leftChunks = 1 if frame.dataLen % self.__SERVER_RECV_BUF_SIZE > 0 else 0
                chunks = frame.dataLen // self.__SERVER_RECV_BUF_SIZE + leftChunks
                offset = 0
                for idx in range(chunks):
                    length = frame.dataLen - offset if offset + self.__SERVER_RECV_BUF_SIZE > frame.dataLen \
                        else self.__SERVER_RECV_BUF_SIZE
                    asyncio.run_coroutine_threadsafe(self.__writeDataInternal(self.__socketWriter, frame.data, \
                                                                    offset, length), loop).result()
                    offset += length
                endTs = time.time()
                if endTs - beginTs > 0.1:
                    self.__logger.warning(f"[Python] push video frame too long, duration: {1000*(endTs - beginTs)}, \
                                    inter duration: {1000*(interTs - beginTs)}")
                self.__pushVideoFrameCnt = self.__pushVideoFrameCnt + 1
                if endTs - self.__sendVideoStatsTs > 2:
                    self.__logger.info(f"[Python] push video frame cnt: {self.__pushVideoFrameCnt}")
                    self.__sendVideoStatsTs = endTs
            return 0
        else:
            return -1

    
    def SetExternalAudioSource(self, enable:bool, sampleRate:int, channelsPerFrame:int) -> int:
        self.__logger.info(f"[Python] set external audio source {enable}")
        if not isinstance(enable, bool) or not isinstance(sampleRate, int) or not isinstance(channelsPerFrame, int):
            return -1
        self.__pushAudioSampleRate = sampleRate
        self.__pushAudioChannel = channelsPerFrame
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_EXTERNAL_AUDIO_SOURCE,
                "enable": enable,
                "sampleRate": sampleRate,
                "channelsPerFrame": channelsPerFrame
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def ClearDataBufferInternal(self) -> int:
        if self.__socketWriter != None:
            self.__eventHandler.OnPushAudioFrameBufferFull(True)
            self.__eventHandler.OnPushVideoFrameBufferFull(True)
            jobj = {
                self.__CMDSTRING: self.__CMD_CLEAR_EXTERNAL_DATA
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__logger.info("[Python] clear data buffer")
            return 0
        else:
            return -1

    def ClearDataBuffer(self) -> int:
        if self.__testLoopbackLatencyEnabled == False:
            return self.ClearDataBufferInternal()
        else:
            return 0

    def GetRemainDataBufferSize(self, dataType:BufferDataType) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_REMAIN_DATA_BUFFER_SIZE,
                "dataType": dataType,
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def GetAudioIsMainSpeaker(self, startTime:int, endTime:int) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_AUDIO_IS_MAIN_SPEAKER,
                "startTime": startTime,
                "endTime": endTime,
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def GetAIVadInfo(self, startTime:int, endTime:int) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_AIVAD_INFO,
                "startTime": startTime,
                "endTime": endTime,
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1
            
    def PushExternalAudioFrameInternal(self, frame:AudioFrameData) -> int:
        beginTs = time.time()
        if not isinstance(frame, AudioFrameData) or not isinstance(frame.data, bytes) or not isinstance(frame.dataLen, int):
            return -1
        if len(frame.data) < frame.dataLen:
            return -1
        if len(frame.data) == 0 or frame.dataLen <= 0:
            return -1

        # if self.__enableDumpPcm and self.__pcmFile != None:
        #     self.__pcmFile.write(audioSamples)

        if self.__socketWriter != None:
            if not self.__pushFirstAudio:
                self.__logger.info(f"[Python] push first audio frame...")
                self.__pushFirstAudio = True
            # lock_address = ctypes.addressof(ctypes.py_object(self.__lock))
            # print(f'Lock address for audio: {lock_address:#x}')
            jobj = {
                self.__CMDSTRING: self.__CMD_PUSH_AUDIO_FRAME,
                "sampleLength": frame.dataLen,
                "timestamp": frame.timeStamp,
                "isEndFrame": frame.isEndFrame,
                "frameID": frame.frameID,
                "sentenceID": frame.sentenceID,
                "sequenceID": frame.sequenceID,
            }
                
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
                interTs = time.time()

                leftChunks = 1 if frame.dataLen % self.__SERVER_RECV_BUF_SIZE > 0 else 0
                chunks = frame.dataLen // self.__SERVER_RECV_BUF_SIZE + leftChunks
                offset = 0
                for idx in range(chunks):
                    length = frame.dataLen - offset if offset + self.__SERVER_RECV_BUF_SIZE > frame.dataLen \
                        else self.__SERVER_RECV_BUF_SIZE
                    asyncio.run_coroutine_threadsafe(self.__writeDataInternal(self.__socketWriter, frame.data,\
                                                                    offset, length), loop).result()
                    offset = offset + length
                
                endTs = time.time()
                if endTs - beginTs > 0.1:
                    self.__logger.warning(f"[Python] push audio frame too long, duration: {1000*(endTs - beginTs)}, \
                                    inter duration: {1000*(interTs - beginTs)}")
                self.__pushAudioFrameCnt = self.__pushAudioFrameCnt + 1
                if endTs - self.__sendAudioStatsTs > 2:
                    self.__logger.info(f"[Python] push audio frame cnt: {self.__pushAudioFrameCnt}")
                    self.__sendAudioStatsTs = endTs
            return 0
        else:
           return -1
        # return -1

    def PushExternalAudioFrameRawData(self, audioSamples:bytes, sampleLength:int, timestamp:int) -> int:
        if self.__testLoopbackLatencyEnabled == False:
            frame = AudioFrameData()
            frame.data = audioSamples
            frame.dataLen = sampleLength
            frame.timeStamp = timestamp
            return self.PushExternalAudioFrameInternal(frame)
        else:
            return 0
    
    def PushExternalAudioFrame(self, frame:AudioFrameData) -> int:
        if self.__testLoopbackLatencyEnabled == False:
            return self.PushExternalAudioFrameInternal(frame)
        else:
            return 0

    def EnableLocalVideo(self, enable:bool) -> int:
        jobj = {
            self.__CMDSTRING: self.__CMD_ENABLE_LOCAL_VIDEO,
            "enable":enable
        }
        if self.__socketWriter != None:
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def SetAudioDelayInfo(self, uid: str, delayInfo: AIAudioQuestionDelay) -> int:
        jobj = {
            self.__CMDSTRING: self.__CMD_SET_AUDIO_DELAY_INFO,
            "uid":uid
        }
        if delayInfo is not None:
            jobjDelayInfo = {
                "ai_question_capture_timestamp": delayInfo.ai_question_capture_timestamp,
                "ai_answer_capture_timestamp": delayInfo.ai_answer_capture_timestamp,
                "ai_sentence_id": delayInfo.ai_sentence_id,
                "ai_client_audio_pub_total_delay": delayInfo.ai_client_audio_pub_total_delay,
                "ai_client_audio_source_cost": delayInfo.ai_client_audio_source_cost,
                "ai_client_audio_mixer_cost": delayInfo.ai_client_audio_mixer_cost,
                "ai_client_audio_encoder_cost": delayInfo.ai_client_audio_encoder_cost,
                "ai_client_audio_netsdk_thr_cost": delayInfo.ai_client_audio_netsdk_thr_cost,
                "ai_client_audio_qos_thr_cost": delayInfo.ai_client_audio_qos_thr_cost,
                "ai_client_audio_pacer_cost": delayInfo.ai_client_audio_pacer_cost,
                "ai_client_up_half_rtt": delayInfo.ai_client_up_half_rtt,
                "ai_linux_down_half_rtt": delayInfo.ai_linux_down_half_rtt,
                "ai_linux_audio_sub_total_delay": delayInfo.ai_linux_audio_sub_total_delay,
                "ai_linux_audio_receive_cost": delayInfo.ai_linux_audio_receive_cost,
                "ai_linux_audio_neteq_cost": delayInfo.ai_linux_audio_neteq_cost,
                "ai_linux_audio_remote_source_cost": delayInfo.ai_linux_audio_remote_source_cost,
                "ai_linux_audio_play_mixer_cost": delayInfo.ai_linux_audio_play_mixer_cost,
                "ai_linux_audio_player_cost": delayInfo.ai_linux_audio_player_cost,
                "ai_agent_audio_asr_cost": delayInfo.ai_agent_audio_asr_cost,
                "ai_agent_audio_llm_cost": delayInfo.ai_agent_audio_llm_cost,
                "ai_agent_audio_tts_cost": delayInfo.ai_agent_audio_tts_cost,
                # "ai_question_agent_begin_timestamp": delayInfo.ai_question_agent_begin_timestamp,
                "ai_agent_audio_total_cost": delayInfo.ai_agent_audio_total_cost,
                "ai_linux_audio_process_smart_denoise_delay": delayInfo.ai_linux_audio_process_smart_denoise_delay
            }
            if delayInfo.ai_linux_grtn_node_delay:
                jobjDelayInfo["ai_linux_grtn_node_delay"] = [
                    {
                        "grtn_node_internal_rtt_half": node.grtn_node_internal_rtt_half,
                        "grtn_node_bef_pacer": node.grtn_node_bef_pacer,
                        "grtn_node_pacer_cost": node.grtn_node_pacer_cost
                    } for node in delayInfo.ai_linux_grtn_node_delay
                ]
            jobj['delayInfo'] = jobjDelayInfo
        else:
            return -1
        if self.__socketWriter != None:
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1


    def EncryptQuestionDelayInfo(self, delayInfo: AIAudioQuestionDelay) -> int:
        jobj = {
            self.__CMDSTRING: self.__CMD_ENCRYPT_DELAY_INFO,
        }
        if delayInfo is not None:
            jobjDelayInfo = {
                "ai_question_capture_timestamp": delayInfo.ai_question_capture_timestamp,
                "ai_answer_capture_timestamp": delayInfo.ai_answer_capture_timestamp,
                "ai_sentence_id": delayInfo.ai_sentence_id,
                "ai_client_audio_pub_total_delay": delayInfo.ai_client_audio_pub_total_delay,
                "ai_client_audio_source_cost": delayInfo.ai_client_audio_source_cost,
                "ai_client_audio_mixer_cost": delayInfo.ai_client_audio_mixer_cost,
                "ai_client_audio_encoder_cost": delayInfo.ai_client_audio_encoder_cost,
                "ai_client_audio_netsdk_thr_cost": delayInfo.ai_client_audio_netsdk_thr_cost,
                "ai_client_audio_qos_thr_cost": delayInfo.ai_client_audio_qos_thr_cost,
                "ai_client_audio_pacer_cost": delayInfo.ai_client_audio_pacer_cost,
                "ai_client_up_half_rtt": delayInfo.ai_client_up_half_rtt,
                "ai_linux_down_half_rtt": delayInfo.ai_linux_down_half_rtt,
                "ai_linux_audio_sub_total_delay": delayInfo.ai_linux_audio_sub_total_delay,
                "ai_linux_audio_receive_cost": delayInfo.ai_linux_audio_receive_cost,
                "ai_linux_audio_neteq_cost": delayInfo.ai_linux_audio_neteq_cost,
                "ai_linux_audio_remote_source_cost": delayInfo.ai_linux_audio_remote_source_cost,
                "ai_linux_audio_play_mixer_cost": delayInfo.ai_linux_audio_play_mixer_cost,
                "ai_linux_audio_player_cost": delayInfo.ai_linux_audio_player_cost,
                "ai_agent_audio_asr_cost": delayInfo.ai_agent_audio_asr_cost,
                "ai_agent_audio_llm_cost": delayInfo.ai_agent_audio_llm_cost,
                "ai_agent_audio_tts_cost": delayInfo.ai_agent_audio_tts_cost,
                # "ai_question_agent_begin_timestamp": delayInfo.ai_question_agent_begin_timestamp,
                "ai_agent_audio_total_cost": delayInfo.ai_agent_audio_total_cost,
                "ai_linux_audio_process_smart_denoise_delay": delayInfo.ai_linux_audio_process_smart_denoise_delay
            }
            if delayInfo.ai_linux_grtn_node_delay:
                jobjDelayInfo["ai_linux_grtn_node_delay"] = [
                    {
                        "grtn_node_internal_rtt_half": node.grtn_node_internal_rtt_half,
                        "grtn_node_bef_pacer": node.grtn_node_bef_pacer,
                        "grtn_node_pacer_cost": node.grtn_node_pacer_cost
                    } for node in delayInfo.ai_linux_grtn_node_delay
                ]
            jobj['delayInfo'] = jobjDelayInfo
        else:
            return -1
        if self.__socketWriter != None:
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1


    def SetEncryptAudioDelayInfo(self, uid: str, sentenceId: int, delayInfo: str, extraDelay: AIExtraDelay) -> int:
        if delayInfo is None:
            return -1
        jobj = {
            self.__CMDSTRING: self.__CMD_SET_AUDIO_DELAY_INFO_V2,
            "uid": uid,
            "sentenceId": sentenceId,
            "delayInfo": delayInfo
        }
        if extraDelay is not None:
            jobjExtraDelay = {
                "avatar_render_cost": extraDelay.avatar_render_cost
            }
            jobj["extraDelay"] = jobjExtraDelay
        if self.__socketWriter != None:
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1


    def SetExternalAudioPublishVolume(self, volume:int) -> int:
        if not isinstance(volume, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_EXTERNAL_AUDIO_PUBLISH_VOLUME,
                "volume": volume
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__externalAudioVolume = volume
            self.__logger.info(f"[Python] set audio volume {volume}")
            return 0
        else:
            self.__externalAudioVolume = 100
            return -1

    
    def GetExternalAudioPublishVolume(self) -> int:
        return self.__externalAudioVolume

    
    def SetAudioProfile(self, audioProfile:AudioQualityMode, audioScene:AudioSceneMode) -> int:
        if not isinstance(audioProfile, AudioQualityMode) or not isinstance(audioScene, AudioSceneMode):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_AUDIO_PROFILE,
                "audioProfile": audioProfile.value,
                "audioScene": audioScene.value
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    
    def SendMediaExtensionMsg(self, message:bytes, length:int, repeatCount:int, delay:int, isKeyFrame:bool) -> int:
        if not isinstance(message, bytes) or not isinstance(length, int) or not isinstance(repeatCount, int) \
            or not isinstance(delay, int) or not isinstance(isKeyFrame, bool):
            return -1
        if self.__socketWriter != None:
            # base64Data = base64.b64encode(message)
            jobj = {
                self.__CMDSTRING: self.__CMD_SEND_MEDIA_EXTENSION_MSG,
                # "message": base64Data.decode('utf-8'),
                "repeatCount": repeatCount,
                "delay": delay,
                "isKeyFrame": isKeyFrame,
                "length": len(message)
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
                leftChunks = 1 if len(message) % self.__SERVER_RECV_BUF_SIZE > 0 else 0
                chunks = len(message) // self.__SERVER_RECV_BUF_SIZE + leftChunks
                offset = 0
                for idx in range(chunks):
                    length = len(message) - offset if offset + self.__SERVER_RECV_BUF_SIZE > len(message) \
                        else self.__SERVER_RECV_BUF_SIZE
                    asyncio.run_coroutine_threadsafe(self.__writeDataInternal(self.__socketWriter, message, \
                                                                    offset, length), loop).result()
                    offset += length
            return 0
        else:
            return -1

    
    def PublishLocalDualStream(self, enabled:bool) -> int:
        if not isinstance(enabled, bool):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PUBLISH_LOCAL_DUAL_STREAM,
                "enable": enabled
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__localDualPublishEnabled = enabled
            self.__logger.info(f"[Python] publish local dual stream {enabled}")
            return 0
        else:
            self.__localDualPublishEnabled = False
            return -1

    
    def PublishLocalVideoStream(self, enabled:bool) -> int:
        if not isinstance(enabled, bool):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PUBLISH_LOCAL_VIDEO_STREAM,
                "enable": enabled
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__localCameraPublishEnabled = enabled
            self.__logger.info(f"[Python] publish local video stream {enabled}")
            return 0
        else:
            self.__localCameraPublishEnabled = False
            return -1

    
    def PublishLocalAudioStream(self, enabled:bool) -> int:
        if not isinstance(enabled, bool):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PUBLISH_LOCAL_AUDIO_STREAM,
                "enable": enabled
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__localAudioPublishEnabled = enabled
            self.__logger.info(f"[Python] publish local audio stream {enabled}")
            return 0
        else:
            self.__localAudioPublishEnabled = False
            return -1


    def PublishScreenShareStream(self, enabled:bool) -> int:
        if not isinstance(enabled, bool):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PUBLISH_SCREEN_SHARE_STREAM,
                "enable": enabled
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__localScreenPublishEnabled = enabled
            self.__logger.info(f"[Python] publish screen share {enabled}")
            return 0
        else:
            self.__localScreenPublishEnabled = False
            return -1

    
    def SubscribeRemoteAudioStream(self, uid:str, sub:bool, config:AudioObserverManualConfig) -> int:
        if not isinstance(sub, bool) or not isinstance(uid, str) or len(uid)<=0:
            return -1
        if self.__socketWriter != None:
            needTestLoopbackLatency = self.NeedTestLoopbackLatency(uid)
            if needTestLoopbackLatency:
                self.ClearDataBufferInternal()
                with self.__lock:
                    if not self.__thread_running:
                        self.__thread_running = True
                        self.__push_thread = threading.Thread(target=self.AudioPushThread)
                        self.__push_thread.start()
            jobj = {
                self.__CMDSTRING: self.__CMD_SUBSCRIBE_REMOTE_AUDIO_STREAM,
                "uid": uid,
                "sub": sub,
                "denoise": config.enableDenoise,
                "voiceprint": config.enableVoiceprintRecognize,
                "asrvad": config.enableAsrCallback,
                "bind": config.isBindPublishLoopDelay
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    
    def SubscribeRemoteVideoStream(self, uid:str, videoTrack:VideoTrack, sub:bool) -> int:
        if not isinstance(sub, bool) or not isinstance(uid, str) or len(uid)<=0 \
            or not isinstance(videoTrack, VideoTrack):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SUBSCRIBE_REMOTE_VIDEO_STREAM,
                "uid": uid,
                "videoTrack": videoTrack.value,
                "sub": sub
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1
        
    
    def SetVideoEncoderConfiguration(self, config:AliEngineVideoEncoderConfiguration) -> int:
        if not isinstance(config, AliEngineVideoEncoderConfiguration):
            return -1
        if self.__socketWriter != None:
            dimensions = {
                "width": config.dimensions.width,
                "height": config.dimensions.height
            }
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_VIDEO_ENCODER_CONFIGURATION,
                "dimensions": dimensions,
                "frameRate": config.frameRate.value,
                "bitrate": config.bitrate,
                "mirrorMode": config.mirrorMode.value,
                "rotationMode": config.rotationMode.value,
                "orientationMode": config.orientationMode.value,
                "keyFrameInterval": config.keyFrameInterval,
                "minBitrate": config.minBitrate
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1
    
    
    def SetScreenShareEncoderConfiguration(self, config:AliEngineScreenShareEncoderConfiguration) -> int:
        if not isinstance(config, AliEngineScreenShareEncoderConfiguration):
            return -1
        if self.__socketWriter != None:
            dimensions = {
                "width": config.dimensions.width,
                "height": config.dimensions.height
            }
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_SCREEN_SHARE_ENCODER_CONFIGURATION,
                "dimensions": dimensions,
                "frameRate": config.frameRate.value,
                "bitrate": config.bitrate,
                "rotationMode": config.rotationMode.value
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    
    def SetRemoteVideoStreamType(self, uid:str, streamType:AliEngineVideoStreamType) -> int:
        if not isinstance(uid, str) or not isinstance(streamType, AliEngineVideoStreamType):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_REMOTE_VIDEO_STREAM_TYPE,
                "uid": uid,
                "streamType": streamType.value
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1
    
    
    def MuteLocalCamera(self, mute:bool) -> int:
        if not isinstance(mute, bool):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_MUTE_LOCAL_CAMERA,
                "mute": mute
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__logger.info(f"[Python] mute local camera {mute}")
            return 0
        else:
            return -1

    
    def MuteLocalMic(self, mute:bool) -> int:
        if not isinstance(mute, bool):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_MUTE_LOCAL_MIC,
                "mute": mute
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__logger.info(f"[Python] mute local mic {mute}")
            return 0
        else:
            return -1

    
    def SetClientRole(self, clientRole:AliEngineClientRole) -> int:
        if not isinstance(clientRole, AliEngineClientRole):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_CLIENTROLE,
                "role": clientRole.value
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__logger.info(f"[Python] set client role {clientRole.value}")
            return 0
        else:
            return -1

    
    def SetRemoteDefaultVideoStreamType(self, streamType:AliEngineVideoStreamType) -> int:
        if not isinstance(streamType, AliEngineVideoStreamType):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_REMOTE_DEFAULT_VIDEO_STREAM_TYPE,
                "streamType": streamType.value
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    
    def LeaveOnceNoStreamer(self, enable:bool) -> None:
        if not isinstance(enable, bool):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_LEAVE_ONCE_NO_STREAMER,
                "enable": enable
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    
    def SetPeriodForCheckPeople(self, seconds:int) -> None:
        if not isinstance(seconds, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_PERIOD_FOR_CHECK_PEOPLE,
                "seconds": seconds
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1


    def SetVideoCallbackPeriod(self, period:int) -> None:
        if not isinstance(period, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_VIDEO_CALLBACK_PERIOD,
                "period": period
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1
    

    def SetParameter(self, params:str) -> int:
        if not isinstance(params, str):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_PARAMETER,
                "param": params
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            self.__logger.info("[Python] set parameter {params}")
            return 0
        else:
            return -1

    
    def GenerateToken(self, authInfo:AuthInfo, appkey:str) -> str:
        combined_str = authInfo.appid + appkey + authInfo.channel + authInfo.userid + str(authInfo.timestamp)
        sha256Token = hashlib.sha256(combined_str.encode()).hexdigest()

        jobj = {
            "appid": authInfo.appid,
            "channelid": authInfo.channel,
            "userid": authInfo.userid,
            "nonce": '',
            "timestamp": authInfo.timestamp,
            "token": sha256Token
        }
        jobj_str = json.dumps(jobj)
        base64Token = base64.b64encode(jobj_str.encode())
        return base64Token.decode('utf-8')


    def SendDataChannelMessage(self, ctrlMsg:AliEngineDataChannelMsg) -> int:
        if not isinstance(ctrlMsg, AliEngineDataChannelMsg):
            return -1
        if self.__socketWriter != None:
            # base64Data = base64.b64encode(ctrlMsg.data)
            # lock_address = ctypes.addressof(ctypes.py_object(self.__lock))
            # print(f'Lock address for data channel: {lock_address:#x}')
            jobj = {
                self.__CMDSTRING: self.__CMD_SEND_DATA_CHANNEL_MSG,
                # "data": base64Data.decode('utf-8'),
                "networkTime": ctrlMsg.networkTime,
                "progress": ctrlMsg.progress,
                "type": ctrlMsg.type.value,
                "dataLen": len(ctrlMsg.data)
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
                leftChunks = 1 if len(ctrlMsg.data) % self.__SERVER_RECV_BUF_SIZE > 0 else 0
                chunks = len(ctrlMsg.data) // self.__SERVER_RECV_BUF_SIZE + leftChunks
                offset = 0
                for idx in range(chunks):
                    length = len(ctrlMsg.data) - offset if offset + self.__SERVER_RECV_BUF_SIZE > len(ctrlMsg.data) \
                        else self.__SERVER_RECV_BUF_SIZE
                    asyncio.run_coroutine_threadsafe(self.__writeDataInternal(self.__socketWriter, ctrlMsg.data, \
                                                                    offset, length), loop).result()
                    offset += length
            return 0
        else:
            return -1
    
    
    def SetVoiceprintVector(self, vector:bytes) -> int:
        if vector != None and len(vector) != 192*4:
            self.__logger.error(f'[Python] Invalid voiceprint vector, length: {len(vector)}')
            return -1
        if self.__socketWriter != None:
            vector_base64 = base64.b64encode(vector).decode('utf-8') if vector != None else 'Null'
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_VOICEPRINT_VECTOR,
                "vector": vector_base64
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    
    def GetVoiceprintVector(self) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_VOICEPRINT_VECTOR
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def NeedTestLoopbackLatency(self, uid:str) -> bool:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_NEED_TEST_CYCLE_ROUND,
                "uid":uid
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()

                async def internalSleep(seconds: float) -> None:
                    await asyncio.sleep(seconds)
                init = time.time()
                while self.__didGetNeedTestLoopbackLatency == False:
                    asyncio.run_coroutine_threadsafe(internalSleep(0.0001), loop).result()
                    if time.time() - init >= 10:
                        return self.__testLoopbackLatencyEnabled
            return self.__testLoopbackLatencyEnabled
        else:
            return self.__testLoopbackLatencyEnabled

    def AudioPushThread(self):
        while self.__thread_running:
            try:
                if not self.__audio_queue.empty() and not self.__pushAudioFull:
                    data = self.__audio_queue.get()
                    loop = self._artc_loop
                    if (self.__remoteAudioSampleRate != 0 and self.__remoteAudioChannel != 0) and (self.__pushAudioSampleRate != self.__remoteAudioSampleRate or self.__pushAudioChannel != self.__remoteAudioChannel):
                        print(f"[Python] ReSetExternalAudioSource: {self.__remoteAudioSampleRate} {self.__remoteAudioChannel}")
                        self.SetExternalAudioSource(True, self.__remoteAudioSampleRate, self.__remoteAudioChannel)
                    else:
                        frame = AudioFrameData()
                        frame.data = data
                        frame.dataLen = len(data)
                        frame.timestamp = 0
                        self.PushExternalAudioFrameInternal(frame)
                else:
                    time.sleep(0.001)
            except Exception as e:
                self.__logger.error(f"AudioPushThread error: {e}")
                break
    
    def GetSDKVersion(self) -> str:
        if len(self.__sdkVersion) > 0:
            return self.__sdkVersion
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_SDK_VERSION,
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
                async def internalSleep(seconds: float) -> None:
                    await asyncio.sleep(seconds)
                init = time.time()
                while len(self.__sdkVersion) == 0:
                    asyncio.run_coroutine_threadsafe(internalSleep(0.0001), loop).result()
                    if time.time() - init >= 10:
                        return self.__sdkVersion
        return self.__sdkVersion

    def GetAudioDumpPath(self) -> str:
        return self.__pushAudioDumpPath

    def PreloadAudioEffect(self, soundId:int, filePath:str) -> int:
        self.__logger.info(f"[Python] PreloadAudioEffect soundId:{soundId} filePath:{filePath}")
        if not isinstance(soundId, int) or not isinstance(filePath, str):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PRELOAD_AUDIO_EFFECT,
                "soundId": soundId,
                "filePath": filePath
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def UnloadAudioEffect(self, soundId:int) -> int:
        self.__logger.info(f"[Python] UnloadAudioEffect soundId:{soundId}")
        if not isinstance(soundId, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_UNLOAD_AUDIO_EFFECT,
                "soundId": soundId
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def PlayAudioEffect(self, soundId:int, filePath:str, config:AliRtcAudioEffectConfig) -> int:
        self.__logger.info(f"[Python] PlayAudioEffect soundId:{soundId} filePath:{filePath}")
        if not isinstance(soundId, int) or not isinstance(filePath, str) or not isinstance(config, AliRtcAudioEffectConfig):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PLAY_AUDIO_EFFECT,
                "soundId": soundId,
                "filePath": filePath,
                "needPublish": config.needPublish,
                "loopCycles": config.loopCycles,
                "startPosMs": config.startPosMs,
                "publishVolume": config.publishVolume,
                "playoutVolume": config.playoutVolume
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def StopAudioEffect(self, soundId:int) -> int:
        self.__logger.info(f"[Python] StopAudioEffect soundId:{soundId}")
        if not isinstance(soundId, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_STOP_AUDIO_EFFECT,
                "soundId": soundId
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def StopAllAudioEffects(self) -> int:
        self.__logger.info(f"[Python] StopAllAudioEffects")
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_STOP_ALL_AUDIO_EFFECTS
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def SetAudioEffectPublishVolume(self, soundId:int, volume:int) -> int:
        self.__logger.info(f"[Python] SetAudioEffectPublishVolume soundId:{soundId} volume:{volume}")
        if not isinstance(soundId, int) or not isinstance(volume, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_AUDIO_EFFECT_PUBLISH_VOLUME,
                "soundId": soundId,
                "volume": volume
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def GetAudioEffectPublishVolume(self, soundId:int) -> int:
        self.__logger.info(f"[Python] GetAudioEffectPublishVolume soundId:{soundId}")
        if not isinstance(soundId, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_AUDIO_EFFECT_PUBLISH_VOLUME,
                "soundId": soundId
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
                async def internalSleep(seconds: float) -> None:
                    await asyncio.sleep(seconds)
                init = time.time()
                while self.__didGetAudioEffectPublishVolume == False:
                    asyncio.run_coroutine_threadsafe(internalSleep(0.0001), loop).result()
                    if time.time() - init >= 10:
                        return 0
            return self.__audioEffectPublishVolume
        else:
            return -1

    def SetAudioEffectPlayoutVolume(self, soundId:int, volume:int) -> int:
        self.__logger.info(f"[Python] SetAudioEffectPlayoutVolume soundId:{soundId} volume:{volume}")
        if not isinstance(soundId, int) or not isinstance(volume, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_AUDIO_EFFECT_PLAYOUT_VOLUME,
                "soundId": soundId,
                "volume": volume
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def GetAudioEffectPlayoutVolume(self, soundId:int) -> int:
        self.__logger.info(f"[Python] GetAudioEffectPlayoutVolume soundId:{soundId}")
        if not isinstance(soundId, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_GET_AUDIO_EFFECT_PLAYOUT_VOLUME,
                "soundId": soundId
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
                async def internalSleep(seconds: float) -> None:
                    await asyncio.sleep(seconds)
                init = time.time()
                while self.__didGetAudioEffectPlayoutVolume == False:
                    asyncio.run_coroutine_threadsafe(internalSleep(0.0001), loop).result()
                    if time.time() - init >= 10:
                        return 0
            return self.__audioEffectPlayoutVolume
        else:
            return -1

    def SetAllAudioEffectsPlayoutVolume(self, volume:int) -> int:
        self.__logger.info(f"[Python] SetAllAudioEffectsPlayoutVolume volume:{volume}")
        if not isinstance(volume, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_ALL_AUDIO_EFFECTS_PLAYOUT_VOLUME,
                "volume": volume
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def SetAllAudioEffectsPublishVolume(self, volume:int) -> int:
        self.__logger.info(f"[Python] SetAllAudioEffectsPublishVolume volume:{volume}")
        if not isinstance(volume, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_SET_ALL_AUDIO_EFFECTS_PUBLISH_VOLUME,
                "volume": volume
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def PauseAudioEffect(self, soundId:int) -> int:
        self.__logger.info(f"[Python] PauseAudioEffect soundId:{soundId}")
        if not isinstance(soundId, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PAUSE_AUDIO_EFFECT,
                "soundId": soundId
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def PauseAllAudioEffects(self) -> int:
        self.__logger.info(f"[Python] PauseAllAudioEffects")
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PAUSE_ALL_AUDIO_EFFECTS
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def ResumeAudioEffect(self, soundId:int) -> int:
        self.__logger.info(f"[Python] ResumeAudioEffect soundId:{soundId}")
        if not isinstance(soundId, int):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_RESUME_AUDIO_EFFECT,
                "soundId": soundId
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def ResumeAllAudioEffects(self) -> int:
        self.__logger.info(f"[Python] ResumeAllAudioEffects")
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_RESUME_ALL_AUDIO_EFFECTS
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def Dial(self, info:AliEngineDialInfo, cfg:AliEngineVoipConfig, config:AudioObserverManualConfig) -> int:
        if not isinstance(info, AliEngineDialInfo) or not isinstance(cfg, AliEngineVoipConfig):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_DIAL,
                "sipDomain": info.sipDomain,
                "sipToken": info.sipToken,
                "callerNumber": info.callerNumber,
                "calleeNumber": info.calleeNumber,
                "calleePrefix": info.calleePrefix,
                "localPublicIp": info.localPublicIp,
                "minPort": info.minPort,
                "maxPort": info.maxPort,
                "sipHeaderExtra": info.sipHeaderExtra,
                "audioCodec": cfg.audioCodec,
                "audioSampleRate": cfg.audioSampleRate,
                "denoise": config.enableDenoise,
                "voiceprint": config.enableVoiceprintRecognize,
                "asrvad": config.enableAsrCallback,
                "ttsvad": config.enableTtsVad,
                "bind": config.isBindPublishLoopDelay
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def DialUpdate(self, cfg:AliEngineVoipConfig) -> int:
        if not isinstance(cfg, AliEngineVoipConfig):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_DIAL_UPDATE,
                "audioCodec": cfg.audioCodec,
                "audioSampleRate": cfg.audioSampleRate
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def HangUp(self) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_HANGUP
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def PickupIncomingCall(self, info:AliEngineIncomingCallInfo, cfg:AliEngineVoipConfig, callbackCfg:AudioObserverManualConfig) -> int:
        if not isinstance(info, AliEngineIncomingCallInfo) or not isinstance(cfg, AliEngineVoipConfig):
            return -1
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_PICKUP_INCOMING_CALL,
                "sipDomain": info.sipDomain,
                "callerNumber": info.callerNumber,
                "calleeNumber": info.calleeNumber,
                "remotePublicIp": info.remotePublicIp,
                "remoteAudioPort": info.remoteAudioPort,
                "minPort": info.minPort,
                "maxPort": info.maxPort,
                "inviteSDPInfo": info.inviteSDPInfo,
                "audioCodec": cfg.audioCodec,
                "audioSampleRate": cfg.audioSampleRate,
                "denoise": callbackCfg.enableDenoise,
                "voiceprint": callbackCfg.enableVoiceprintRecognize,
                "asrvad": callbackCfg.enableAsrCallback,
                "ttsvad": callbackCfg.enableTtsVad,
                "bind": callbackCfg.isBindPublishLoopDelay
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def ConnectIncomingCall(self) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_CONNECT_INCOMING_CALL
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def DisConnectIncomingCall(self) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_DIS_CONNECT_INCOMING_CALL
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1

    def EnableAudioFrameObserver(self, enabled:bool, audioSource:AliRtcAudioSource, config:AliRtcAudioFrameObserverConfig) -> int:
        if self.__socketWriter != None:
            jobj = {
                self.__CMDSTRING: self.__CMD_ENABLE_AUDIO_FRMAE_OBSERVER,
                "enabled": enabled,
                "audioSource": audioSource.value,
                "sampleRate": config.sampleRate.value,
                "channels": config.channels.value
            }
            jobj_str = json.dumps(jobj)
            with self.__lock:
                loop = self._artc_loop
                asyncio.run_coroutine_threadsafe(self.__writeData(self.__socketWriter, jobj_str.encode('utf-8')), loop).result()
            return 0
        else:
            return -1
