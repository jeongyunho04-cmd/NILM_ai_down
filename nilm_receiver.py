#!/usr/bin/env python3
"""
NILM 수신기 (노트북 쪽) - 프로토콜 v4
=====================================
STM32 -> ESP-01S 가 TCP 클라이언트로 접속해 오는 것을 받아 CSV로 저장한다.
펌웨어 NILM_ECE_IF/Core/Inc/nilm_link.h 의 프레임 형식과 1:1로 맞춰져 있다.

프레임 (전부 리틀엔디언, 총 3213바이트):

    [A5 5A] [ver=04] [len=3206]   공통부 86B   주기별 104B x 30 = 3120B  [CRC16]
    └─ 헤더 5B ─────────────────┘              └─ cycles[0] .. cycles[29] ┘  2B

    CRC16-CCITT(다항식 0x1021, init 0xFFFF), ver 바이트부터 페이로드 끝까지.

ACK (수신기 -> 보드, 5바이트): [06] [seq u32 LE]
    페이로드 첫 필드인 seq를 그대로 되돌려 준다. v3의 ACK는 0x06 한
    바이트여서 어느 프레임에 대한 응답인지 알 수 없었다 - 자세한 사정은
    frame_stream() 과 ack_for() 의 주석에 있다.
    v3 펌웨어와는 호환되지 않는다. 양쪽을 같이 올려야 한다.

  - 공통부(SLOW) : 0.5초 평균. 주파수 / Vrms / 전압 THD·고조파 / 품질 플래그.
                   over_cycle_map 은 측정범위 초과(두 전류 경로 모두 클리핑)가
                   발생한 주기를 표시하는 30비트 비트맵.
  - 주기별(CYCLE): 계통 1주기(1/60초)마다의 전류 고조파 15차 rms/위상, P,
                   Irms, V-I 위상차, 범위초과 샘플 수. 평균하지 않은 원본.

CSV는 "주기 1개 = 1행"으로 쓴다(초당 60행). 공통부 값은 그 주기가 속한
0.5초 창의 값으로 각 행에 반복해 넣는다 - pandas로 바로 읽어 쓰기 위함.

CSV는 이 스크립트 옆의 data/ 폴더에 쌓인다(없으면 만든다). 실행 디렉터리와
무관하게 항상 같은 곳이라 학습 데이터를 한 군데에서 찾을 수 있다.

사용법:
    python nilm_receiver.py                     # data/nilm_<날짜시각>.csv
    python nilm_receiver.py --csv run1.csv      # data/run1.csv (있으면 이어쓰기)
    python nilm_receiver.py --csv D:\\tmp\\x.csv  # 경로를 주면 그대로
    python nilm_receiver.py --no-csv            # 콘솔만
    python nilm_receiver.py --port 5000         # 포트 변경 (기본 5000)
    python nilm_receiver.py --quiet             # 콘솔 요약 끄기

준비 (Windows 설정 -> 네트워크 및 인터넷 -> 모바일 핫스팟):
    1. 핫스팟 이름/암호를 nilm_link.h 의 NILM_WIFI_SSID / NILM_WIFI_PASS 와 일치
    2. 네트워크 대역 2.4GHz (ESP8266은 5GHz를 못 잡는다)
    3. "전원 절약" 끄기, 측정 중 노트북 절전 금지
    4. 방화벽에서 이 포트의 인바운드 TCP 허용 (아래 명령 참고)
       netsh advfirewall firewall add rule name="NILM 5000" ^
             dir=in action=allow protocol=TCP localport=5000
실행 순서: 핫스팟 ON -> 이 스크립트 실행 -> 보드 리셋
"""
import argparse
import csv
import os
import select
import socket
import struct
import sys
import time

# ── 프로토콜 상수 (펌웨어 nilm_link.h 와 반드시 일치) ────────────────────────
MAGIC = b"\xa5\x5a"
PROTO_VER = 4
ACK_BYTE = b"\x06"     # ACK 첫 바이트. 뒤에 그 프레임의 seq(u32 LE)가 붙는다


def ack_for(seq: int) -> bytes:
    """ACK 5바이트: 0x06 + seq(u32 LE).

    seq를 실어 보내야 보드가 "어느 프레임에 대한 응답인지" 알 수 있다.
    v3에서는 0x06 한 바이트뿐이라 구분이 안 됐고, 그래서 중복 프레임에는
    ACK를 아예 못 줬다(주면 보드가 다음 프레임의 ACK로 오해했다).
    그 결과 ACK 한 장만 유실돼도 보드는 오지 않을 응답을 기다리며 재전송
    한도 12.5초를 태웠고, 그동안 큐가 넘쳐 프레임이 사라졌다.
    """
    return ACK_BYTE + struct.pack("<I", seq)


# 프레임이 안 와도 이 주기로 ACK 를 한 번 내보낸다(하트비트).
HEARTBEAT_S = 0.25

# seq 가 이 폭 안에서 뒤로 오는 것은 유실이 아니라 순서 뒤바뀜으로 본다.
#
# 펌웨어가 송신 윈도우를 쓰면 새 프레임을 계속 내보내는 동안 확인 못 받은
# 옛 장을 골라 재전송한다(선택적 재전송). 그래서 37 -> 28 -> 38 -> 29 처럼
# 들어온다. 28 은 앞서 죽었던 것이 뒤늦게 복구된 것이라 유실이 아니라 오히려
# 회수된 데이터다. 이 폭보다 더 뒤로 가야 진짜 보드 리셋으로 판정한다.
# 보드 큐(최대 40장)보다 넉넉히 잡는다.
#
# 크게 잡을수록 안전하지만 대가가 있다: 유실은 seq 가 이만큼 지나가야 확정되므로
# 그만큼(x0.5초) 늦게 보고된다. 128 로 뒀더니 64초 뒤에야 로그가 떠서 펌웨어
# 로그와 나란히 볼 수가 없었다. 실제 뒤바뀜 폭은 펌웨어 송신 윈도우를 넘지
# 않으므로 32면 그 두 배로 넉넉하고 보고 지연은 16초로 줄어든다.
REORDER_MAX = 32


def send_ack(conn, seq: int, stats: dict, wait: bool = True) -> bool:
    """ACK 한 장을 보낸다. 계속 진행해도 되면 True, 소켓이 죽었으면 False.

    송신 지연은 연결을 끊을 이유가 못 된다. ACK 를 한 장 못 보내도 보드가
    재전송으로 복구한다. 예전에는 여기서 socket.timeout(OSError 의 하위
    클래스)을 그대로 삼켜 로그 한 줄 없이 연결을 버렸고, 그러면 보드가
    재접속에 20초를 쓰는 동안 큐가 넘쳐 데이터를 잃었다.

    wait=False (하트비트) 는 지금 당장 쓸 수 있을 때만 보내고 아니면 건너뛴다.
    여기서 블록되면 하트비트 주기 자체가 무너지는데, 하필 그 일이 일어나는
    구간이 우리가 재려던 "하향이 좁아졌나" 바로 그 구간이다. 즉 블로킹
    하트비트는 측정 대상을 측정 도구가 망가뜨리는 꼴이라, rx 가 낮은 것이
    경로 탓인지 우리가 덜 보낸 탓인지 구분할 수 없게 만든다.
    """
    if not wait:
        try:
            if not select.select([], [conn], [], 0)[1]:
                stats["ack_blocked"] += 1
                return True            # 소켓이 막혀 있다 - 이번 건 건너뛴다
        except OSError:
            return True
    try:
        conn.sendall(ack_for(seq))
        stats["ack_sent"] += 1
    except socket.timeout:
        stats["ack_slow"] += 1
        print(f"[!] ACK 송신이 1초 안에 안 나감 (seq {seq})"
              f" - 역방향(PC->보드)이 막혀 있습니다")
    except OSError as e:
        print(f"[-] ACK 송신 실패: {e}")
        return False
    return True
HARMONICS = 15
CYCLES = 30
CYCLE_HZ = 60.0        # 주기별 데이터의 시간 해상도

# NILM_WireSlow: seq, freq, vrms, thd_v, vh[15], over_range, clip_volt,
#                range, flags, over_map
SLOW_FMT = "<I3f15fHHBBI"
SLOW_SIZE = struct.calcsize(SLOW_FMT)            # = 86

# NILM_WireCycle: irms, p_w, phase_cdeg, range, over_count,
#                 ih[15], ih_cdeg[15], reserved
CYC_FMT = "<ffhBB15f15hH"
CYC_SIZE = struct.calcsize(CYC_FMT)              # = 104

PAYLOAD_SIZE = SLOW_SIZE + CYCLES * CYC_SIZE     # = 3206
FRAME_SIZE = 5 + PAYLOAD_SIZE + 2                # = 3213

assert SLOW_SIZE == 86, f"SLOW_SIZE={SLOW_SIZE}, 펌웨어와 어긋남"
assert CYC_SIZE == 104, f"CYC_SIZE={CYC_SIZE}, 펌웨어와 어긋남"

# 이 시간 동안 무수신이면 죽은 연결로 보고 끊는다.
# 펌웨어는 ACK가 안 오면 같은 프레임을 2.5초 간격으로 최대 5번까지 다시
# 보낸다(nilm_link.h). 그동안 WiFi가 정말 막혀 있으면 이쪽엔 아무것도 안
# 들어오는데, 이 값이 짧으면 곧 되살아날 연결을 먼저 끊어 버린다.
# 실제로 5초로 두었더니 9초짜리 정체에서 연결이 끊기고 ESP 재접속까지
# 겹쳐 공백이 더 커졌다. 펌웨어가 한 프레임에 쓰는 12.5초보다 길게 잡는다.
#
# 이 값은 반드시 펌웨어의 NILM_LINK_DEAD_MS(20초)보다 **작아야** 한다.
# 보드가 재접속을 걸기 전에 낡은 소켓을 비워 둬야 그 연결을 곧바로 받는다.
#
#   12.5초  보드가 한 프레임에 쓰는 재전송 한도 (이보다는 커야 한다.
#           안 그러면 되살아날 연결을 먼저 끊는다 - 5초로 뒀다가 9초짜리
#           정체에서 그렇게 깨진 적이 있다)
#   15.0초  <- 이 값. 낡은 소켓을 놓는다
#   20.0초  보드가 링크를 포기하고 재접속 (수신기는 이미 accept 대기 중)
#
# 30초로 올렸다가 되돌렸다. DEAD_MS 보다 크게 잡으면 보드가 20초에 건 새
# 연결이 백로그에서 10초를 굶다가 보드에게 다시 버려지고(RST), 수신기는
# 뒤늦게 그것을 accept 해 2프레임만 읽은 뒤 WinError 10054 를 맞는다.
# 실측에서 "2프레임짜리 연결 + RST"가 그렇게 반복됐다.
#
# 다만 이 순서에 기대는 것 자체가 약하다. 실제 판단은 frame_stream() 에서
# "새 연결이 대기 중인가"로 하고, 이 타이머는 보드가 재접속조차 못 하는
# 경우(핫스팟 꺼짐 등)를 위한 보루로만 쓴다.
STALE_TIMEOUT = 15.0


# ── CRC16-CCITT (0x1021, init 0xFFFF) ───────────────────────────────────────
def _make_crc_table():
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
        table.append(crc)
    return table


_CRC_TABLE = _make_crc_table()


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[(crc >> 8) ^ byte]
    return crc


# ── 페이로드 -> 파이썬 dict ─────────────────────────────────────────────────
def parse_payload(payload: bytes) -> dict:
    s = struct.unpack(SLOW_FMT, payload[:SLOW_SIZE])
    d = {
        "seq": s[0],
        "freq_hz": s[1],
        "vrms": s[2],
        "thd_v": s[3],
        "vh_rms": s[4:4 + HARMONICS],
        "over_range": s[4 + HARMONICS],
        "clip_volt": s[5 + HARMONICS],
        "range": s[6 + HARMONICS],
        "flags": s[7 + HARMONICS],
        "over_map": s[8 + HARMONICS],
    }
    d["pll_locked"] = bool(d["flags"] & 0x01)
    d["cal_applied"] = bool(d["flags"] & 0x02)
    # 비트맵 -> 측정범위 초과가 발생한 주기 인덱스 목록
    d["over_cycles"] = [i for i in range(CYCLES) if d["over_map"] & (1 << i)]

    cycles = []
    off = SLOW_SIZE
    for _ in range(CYCLES):
        c = struct.unpack(CYC_FMT, payload[off:off + CYC_SIZE])
        cycles.append({
            "irms": c[0],
            "p_w": c[1],
            "phase_deg": c[2] / 100.0,          # 0.01도 -> 도
            "range": c[3],
            "over": c[4],
            "ih_rms": c[5:5 + HARMONICS],
            "ih_deg": [x / 100.0 for x in c[5 + HARMONICS:5 + 2 * HARMONICS]],
        })
        off += CYC_SIZE
    d["cycles"] = cycles
    return d


# ── 소켓 스트림 -> 프레임 ───────────────────────────────────────────────────
def frame_stream(conn: socket.socket, stats: dict, seen: set, srv=None):
    """바이트 스트림에서 CRC까지 통과한 프레임의 payload만 뽑아 yield.

    seen 은 이미 받은 seq 집합이다. 펌웨어는 ACK가 제때 안 오면 같은
    프레임을 다시 보내는데, 첫 장이 무사히 갔고 ACK만 늦었던 경우엔 같은
    seq가 두 번 도착한다.

    v4부터 ACK에 seq가 실리므로 중복에도 그대로 ACK를 돌려준다. 보드는
    seq를 보고 "지금 기다리는 장"일 때만 큐를 전진시키고, 이미 뺀 장에
    대한 ACK는 생존 신호로만 쓴다 - v3에서 중복에 ACK를 못 주던 이유였던
    오해가 구조적으로 불가능해졌다.

    중복에 ACK를 주는 것이 오히려 중요하다. 원본은 도착했는데 ACK만
    유실된 경우, v3에서는 보드가 영영 오지 않을 응답을 기다리며 재전송
    한도(2.5초 x 5 = 12.5초)를 전부 태웠고 그 사이 큐가 넘쳐 프레임이
    사라졌다. 이제는 재전송 한 번으로 즉시 복구된다.
    중복 프레임 자체는 CSV에 두 번 쓰지 않도록 yield 하지 않는다.       """
    buf = b""
    # settimeout 은 recv 와 send 에 함께 걸린다. 대기 주기는 select 로 따로
    # 잡고, 이 값은 sendall 이 얼마나 버틸지로만 쓴다.
    conn.settimeout(1.0)
    t_last = time.time()      # 마지막으로 데이터가 온 시각
    t_ack = 0.0               # 마지막으로 ACK 를 내보낸 시각
    last_seq = None           # 하트비트에 실어 보낼 seq
    # 10초마다 한 줄. 펌웨어 로그도 10초 주기라 나란히 놓고 비교하려는 것이다:
    #   보낸 ACK x 5바이트  vs  펌웨어 rx 증가분  ->  하향 도달률
    t_report = time.time()
    snap = dict(stats)
    while True:
        now_r = time.time()
        if now_r - t_report >= 10.0:
            d_sent = stats["ack_sent"] - snap["ack_sent"]
            d_blk = stats["ack_blocked"] - snap["ack_blocked"]
            d_slow = stats["ack_slow"] - snap["ack_slow"]
            print(f"[=] 10초: ACK 송신 {d_sent}개({d_sent * 5}바이트)"
                  f"  건너뜀 {d_blk}  지연 {d_slow}"
                  f"   <- 펌웨어 rx 증가분과 비교하세요")
            t_report = now_r
            snap = dict(stats)
        # 데이터와 "새 연결 대기"를 한 번에 기다린다.
        #
        # 새 연결 감지: 보드는 한 번에 하나만 연결하므로, 듣는 소켓에 새
        # 연결이 대기 중이라는 것 자체가 "지금 붙들고 있는 이것은 이미
        # 죽었다"는 증거다. 시간이 아니라 사실로 판단한다.
        # 타임아웃으로만 판단하던 때는 순서가 어긋나면 바로 깨졌다:
        # STALE_TIMEOUT(30초) > 보드 DEAD_MS(20초) 로 뒀더니 보드가 20초에
        # 건 새 연결이 백로그에서 10초를 굶다가 보드에게 버려졌고(RST),
        # 수신기는 뒤늦게 그것을 accept 해 2프레임만 읽고 WinError 10054 를
        # 맞았다 - "2프레임짜리 연결"이 반복된 것이 이것이다.
        watch = [conn] if srv is None else [conn, srv]
        try:
            ready = select.select(watch, [], [], HEARTBEAT_S)[0]
        except OSError as e:
            print(f"[-] select 오류: {e}")
            return
        if srv is not None and srv in ready:
            print("[-] 새 연결이 대기 중입니다 - 낡은 소켓을 즉시 버립니다.")
            return

        if conn not in ready:
            # --- 하트비트 ACK -------------------------------------------
            # 프레임이 안 와도 주기적으로 ACK 를 내보낸다. 보드는 seq 가
            # 안 맞으니 큐를 전진시키지 않고 생존 신호로만 쓴다.
            #
            # 목적은 두 가지다. 첫째, 보드의 rx 카운터가 하향 경로의 순수한
            # 측정치가 된다 - 지금은 "ACK 가 막혔나"와 "프레임이 안 나가서
            # ACK 생길 일이 없나"가 섞여 있어 구분이 안 된다. 일정 주기로
            # 계속 보내면 rx=0 이 곧 "하향이 죽었다"가 된다.
            # 둘째, 간헐적으로 막히는 구간을 뚫고 나갈 기회가 초당 4번으로
            # 늘고, 보드의 생존 감시(s_ack_t0)가 계속 먹여져 헛된 DEAD_MS
            # 발동이 줄어든다.
            now = time.time()
            if last_seq is not None and (now - t_ack) >= HEARTBEAT_S:
                if not send_ack(conn, last_seq, stats, wait=False):
                    return
                t_ack = now
            if now - t_last > STALE_TIMEOUT:
                print(f"[-] {STALE_TIMEOUT:.0f}초간 수신 없음 - 연결을 끊습니다.")
                return
            continue

        try:
            chunk = conn.recv(16384)
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[-] 소켓 오류: {e}")
            return
        if not chunk:
            print("[-] 상대가 연결을 닫았습니다 (FIN)")
            return
        t_last = time.time()
        buf += chunk

        while True:
            idx = buf.find(MAGIC)
            if idx < 0:
                # 매직이 없음: 매직이 경계에 걸쳤을 수 있으니 1바이트만 남긴다
                if len(buf) > 1:
                    stats["resync"] += len(buf) - 1
                buf = buf[-1:]
                break
            if idx > 0:
                stats["resync"] += idx
                buf = buf[idx:]             # 매직 앞 쓰레기 제거
            if len(buf) < 5:
                break                       # 헤더가 아직 다 안 옴

            ver = buf[2]
            length = buf[3] | (buf[4] << 8)
            if ver != PROTO_VER or length != PAYLOAD_SIZE:
                # 형식 불일치 = 데이터 안의 우연한 A5 5A. 다음 매직부터 재탐색
                stats["resync"] += 2
                buf = buf[2:]
                continue

            total = 5 + length + 2
            if len(buf) < total:
                break                       # 본문이 아직 다 안 옴

            crc_rx = buf[total - 2] | (buf[total - 1] << 8)
            if crc16_ccitt(buf[2:total - 2]) != crc_rx:
                # CRC 실패. 여기서 total(3213)바이트를 통째로 버리면 안 된다.
                #
                # 손상의 실제 형태는 "프레임 중간에서 바이트가 사라짐"이다
                # (STM32->ESP UART에는 무결성 보호가 없다. TCP는 깨진 바이트를
                #  올려보내지 않으므로, 여기서 CRC가 틀린다는 것 자체가 손상이
                #  ESP의 TCP 스택에 들어가기 전에 났다는 뜻이다).
                # 바이트가 빠지면 헤더는 멀쩡해 LEN 검증을 통과하지만 프레임
                # 끝이 total보다 앞으로 당겨진다. 그대로 total을 소비하면
                # **다음 프레임의 머리까지 먹어 들어가** 멀쩡한 장까지 죽는다.
                # 실측 74분에서 CRC오류 37 + 재동기 24프레임분 = 유실 87의 79%가
                # 이 증폭으로 설명됐다.
                #
                # 그래서 매직 2바이트만 넘기고 다시 찾는다. 빠진 바이트만큼
                # 앞으로 당겨진 다음 프레임의 매직을 이 안에서 되찾을 수 있다.
                # 손상이 단순 비트 반전이어서 프레임 길이가 멀쩡한 경우엔 본문을
                # 훑고 지나가며 재동기로 버려지므로 결과가 전과 같다 - 즉 이
                # 변경은 최악의 경우에도 손해가 없다.
                stats["crc_err"] += 1
                stats["resync"] += 2
                buf = buf[2:]
                continue

            frame, buf = buf[:total], buf[total:]

            seq = struct.unpack_from("<I", frame, 5)[0]
            dup = seq in seen
            if dup:
                stats["dup"] += 1
            else:
                seen.add(seq)
                if len(seen) > 4096:        # 오래된 것부터 정리
                    seen.difference_update({s for s in seen if s < seq - 2048})

            # ACK 회신: 펌웨어가 이걸 흐름제어로 쓴다. 이 응답을 받아야
            # 다음 프레임을 내보내므로 절대 빼면 안 된다.
            # 중복에도 반드시 보낸다 - 그래야 유실된 ACK가 복구된다.
            if not send_ack(conn, seq, stats):
                return
            last_seq = seq
            t_ack = time.time()
            if dup:
                continue                    # 데이터는 이미 기록했다
            yield frame[5:-2]


# ── CSV ─────────────────────────────────────────────────────────────────────
CSV_HEADER = (
    ["host_time", "t_s", "seq", "cycle",
     # --- 그 주기의 값 (평균 없음) ---
     "irms", "p_w", "phase_deg", "range", "over_count", "over_range",
     # --- 그 주기가 속한 0.5초 창의 공통 값 ---
     "freq_hz", "vrms", "thd_v", "pll_locked", "cal_applied",
     "win_range", "win_over_range_count", "win_clip_volt_count"]
    + [f"ih{h}" for h in range(1, HARMONICS + 1)]
    + [f"ihdeg{h}" for h in range(1, HARMONICS + 1)]
    + [f"vh{h}" for h in range(1, HARMONICS + 1)]
)


# CSV는 전부 이 폴더 아래에 모은다. 스크립트 위치 기준이라 어느 디렉터리에서
# 실행하든 같은 곳에 쌓이고, 학습 데이터를 한 곳에서 찾을 수 있다.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def resolve_csv_path(arg):
    """--csv 인자를 실제 경로로.

    이름만 주면(예: run1.csv) data/ 안에 넣고, 경로를 포함해 주면
    (예: ../other/x.csv, D:\\tmp\\x.csv) 그 뜻을 그대로 존중한다.
    """
    if arg is None:
        return os.path.join(DATA_DIR, time.strftime("nilm_%Y%m%d_%H%M%S.csv"))
    if os.path.dirname(arg):
        return arg
    return os.path.join(DATA_DIR, arg)


def open_csv(path: str):
    """이어쓰기로 열고, 새 파일이면 헤더를 넣는다."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(CSV_HEADER)
    return f, w


def write_frame(writer, d: dict, t_recv: float, t0_dev):
    """프레임 1개 -> CSV 30행.

    t_s 는 보드 기준 시각이다. seq 는 0.5초마다 1씩 오르므로
        t_s = (seq - seq0) * 0.5 + cycle / 60
    이 되고, 호스트 스케줄링 지터와 무관한 균일 시간축이 된다.
    (호스트 벽시계는 host_time 열에 따로 남긴다)
    """
    host = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_recv)) + \
        f".{int((t_recv % 1) * 1000):03d}"
    base = (d["seq"] - t0_dev) * 0.5
    over_set = set(d["over_cycles"])
    vh = [f"{x:.4f}" for x in d["vh_rms"]]

    for i, c in enumerate(d["cycles"]):
        writer.writerow(
            [host, f"{base + i / CYCLE_HZ:.6f}", d["seq"], i,
             f"{c['irms']:.6f}", f"{c['p_w']:.4f}", f"{c['phase_deg']:.2f}",
             c["range"], c["over"], int(i in over_set),
             f"{d['freq_hz']:.4f}", f"{d['vrms']:.3f}", f"{d['thd_v']:.5f}",
             int(d["pll_locked"]), int(d["cal_applied"]),
             d["range"], d["over_range"], d["clip_volt"]]
            + [f"{x:.6f}" for x in c["ih_rms"]]
            + [f"{x:.2f}" for x in c["ih_deg"]]
            + vh)


# ── 콘솔 요약 ───────────────────────────────────────────────────────────────
def summary_line(d: dict) -> str:
    cyc = d["cycles"]
    p = [c["p_w"] for c in cyc]
    last = cyc[-1]
    msg = (f"#{d['seq']:06d} "
           f"f={d['freq_hz']:7.3f}{'L' if d['pll_locked'] else 'u'} "
           f"V={d['vrms']:6.1f} "
           f"P={sum(p) / len(p):8.2f}W (min {min(p):7.1f} / max {max(p):7.1f}) "
           f"I1={last['ih_rms'][0]:8.4f}A "
           f"ph={last['phase_deg']:7.2f}d "
           f"rng={'HIGH' if d['range'] else 'LOW '}"
           f"{'*' if d['cal_applied'] else ' '}")
    if d["over_cycles"]:
        idx = ",".join(str(i) for i in d["over_cycles"])
        msg += f"  !! OVER-RANGE @cycle[{idx}]"
    return msg


# 유실 원인 분류. 이 두 갈래가 대응이 완전히 다르다.
CAUSE_RECV = "recv"    # 프레임은 왔는데 수신기가 파싱 못 해 버림
CAUSE_BOARD = "board"  # 프레임이 선로에 나오지도 않았다


def classify_loss(delta: dict, dwall: float) -> tuple:
    """유실 하나의 원인을 파서 카운터의 증분으로 가른다.

    유실 직전 구간에서 CRC오류나 재동기가 늘었다면, 그 프레임은 도착했으나
    깨져서 버려진 것이다(바이트 무결성 문제).

    셋 다 그대로인데 seq만 건너뛰었다면 프레임이 선로에 나오지도 않은 것이다.
    여기서 멈춰야 한다 - 수신기는 보드의 큐 상태를 볼 수 없으므로 그 이상은
    단정할 수 없다. 원인은 둘 중 하나이고, 가르려면 펌웨어의 drop 카운터가
    필요하다:
      - 큐 축출     : 링크가 생산 속도를 못 따라가 오래된 것부터 밀려났다.
                      설계된 동작이다(항상 최근 20초를 유지한다).
      - 보드가 잃음 : 보냈다고 여기는데 도착하지 않았다. ACK 처리나 ESP 의심.

    예전에는 이 경우를 "ACK 처리를 의심하라"고 단정했는데, 큐 축출까지 같은
    분류로 묶여서 멀쩡한 동작을 버그로 지목했다.

    dwall(직전 프레임과의 벽시계 간격)은 판정에 쓰지 않고 참고로만 찍는다.
    """
    if delta["crc_err"] > 0 or delta["resync"] > 0:
        return CAUSE_RECV, "수신기 폐기(도착했으나 깨짐)"
    return CAUSE_BOARD, "보드가 안 보냄"


def loss_str(got: int, lost: int) -> str:
    """수신/유실을 '몇 개 중 몇 개(몇 %)' 형태로.

    분모는 seq로 셈한 '와야 했던 프레임 수'(수신 + 공백)다. 보드가 리셋되어
    seq가 되돌아간 구간과 연결이 끊겨 있던 시간은 세지 않는다 - 그 동안의
    공백은 링크 유실이 아니라 그냥 측정이 없던 시간이기 때문이다.
    따라서 이 값은 "붙어 있는 동안 얼마나 흘렸나"를 뜻한다.               """
    expect = got + lost
    if expect == 0:
        return "수신 없음"
    return f"{got}/{expect}프레임, 유실 {lost} ({lost / expect * 100:.2f}%)"


def print_summary(frames: int, lost: int, stats: dict, dur: float, csv_path,
                  cause_lost=None, lost_gap=0):
    """종료 시 최종 집계. 유실률이 이 스크립트의 핵심 품질 지표다.

    유실을 두 갈래로 나눠 보여 준다. 원인도 대응도 다르기 때문이다:
      - 연결 중  : 링크가 붙어 있는데 흘린 것. 바이트 손상이 주범이다.
      - 끊긴 동안: 링크가 죽어 있는 사이 보드 큐가 넘쳐 버린 것.
                   QUEUE_N 을 키우거나 무음 시간을 줄여야 한다.
    예전에는 뒤엣것을 아예 안 셌다. 그래서 실제 10%대 유실이 1%대로 보였다.
    """
    expect = frames + lost + lost_gap
    total_lost = lost + lost_gap
    rate = (total_lost / expect * 100.0) if expect else 0.0

    print(f"\n종료: {dur:.0f}초 동안 {frames}프레임 수신 "
          f"({frames * CYCLES}주기, "
          f"{frames * CYCLES / max(dur, 1e-9):.1f}주기/초)")
    print(f"  유실률 : {rate:.2f}%  (유실 {total_lost} / 기대 {expect}프레임"
          f" = 주기 {total_lost * CYCLES}개 분량)")
    if lost_gap:
        print(f"    - 연결 중  : {lost}프레임")
        print(f"    - 끊긴 동안: {lost_gap}프레임 "
              f"({lost_gap * 0.5:.0f}초 분량, 보드 큐 넘침)")
    print(f"  CRC오류: {stats['crc_err']}프레임"
          f"   중복(재전송): {stats.get('dup', 0)}프레임"
          f"   재동기: {stats['resync']}바이트")
    sent = stats.get("ack_sent", 0)
    if sent:
        print(f"  ACK 송신: {sent}개 = {sent * 5}바이트"
              f"   건너뜀 {stats.get('ack_blocked', 0)}"
              f"   지연 {stats.get('ack_slow', 0)}")
        print(f"    -> 펌웨어의 rx 총량과 비교하세요. rx / {sent * 5} 가"
              f" 하향 도달률입니다.")
        print(f"       도달률이 100%에 가까우면 하향은 멀쩡하고 문제는"
              f" 다른 곳, 낮으면 PC->보드 경로가 실제로 좁은 것입니다.")
    if stats.get("ack_blocked", 0) > sent * 0.1:
        print(f"  ! 하트비트를 {stats['ack_blocked']}회 건너뛰었습니다"
              f" - 소켓이 쓸 수 없는 상태였다는 뜻이라, PC 쪽에서 이미"
              f" 역압을 받고 있었습니다.")

    # 유실을 원인별로 나눠 보여 준다. 대응이 완전히 갈리기 때문이다.
    if cause_lost and lost:
        r = cause_lost.get(CAUSE_RECV, 0)
        b = cause_lost.get(CAUSE_BOARD, 0)
        print(f"  유실 원인: 수신기 폐기 {r}프레임 ({r / lost * 100:.0f}%)"
              f" / 보드가 안 보냄 {b}프레임 ({b / lost * 100:.0f}%)")
        if b > r:
            print("  ! 대부분이 '보드가 안 보냄'입니다. 프레임이 선로에 나오지도"
                  " 않았다는 뜻입니다. 여기서부터는 펌웨어 로그의 drop 카운터와"
                  " 맞춰 봐야 갈립니다:")
            print("      drop 이 같이 올랐다 -> 큐 축출. 링크가 생산 속도를 못"
                  " 따라간 것이고, 설계된 동작입니다. resent 와 무음 구간을"
                  " 보세요.")
            print("      drop 은 그대로다    -> 보드는 보냈다고 여기는데 도착하지"
                  " 않았습니다. ACK 처리나 ESP 쪽을 의심하세요.")
        elif r > 0:
            print("  ! 대부분이 '수신기 폐기'입니다. 프레임은 도착했는데 깨져"
                  " 있었다는 뜻이라, STM32->ESP UART 구간의 바이트 무결성"
                  " 문제입니다(TCP는 깨진 바이트를 올려보내지 않습니다).")
    if rate > 1.0:
        # 유실은 대부분 ESP 버퍼 넘침이다. 보레이트 상향(460800)이 안 됐거나
        # 청크 간격이 짧으면 여기가 가장 먼저 티가 난다.
        print("  ! 유실률이 1%를 넘습니다. 펌웨어 로그에서 baud가 460800으로"
              " 올라갔는지, NILM_LINK_TX_GAP_MS를 늘릴 여지가 있는지"
              " 확인하세요.")
    if csv_path:
        print(f"저장됨: {csv_path}")


def disable_console_quickedit():
    """Windows 콘솔의 '빠른 편집 모드'를 끈다. 되돌리기용 정보를 반환.

    빠른 편집이 켜져 있으면 콘솔 창을 클릭하는 순간 선택 모드로 들어가면서
    프로세스의 출력이 블로킹된다. 즉 측정 중에 창을 실수로 한 번 누르면
    수신이 통째로 멈춘다 - 실측에서 20초짜리 정지가 이것으로 보였다.
    (그때 데이터는 TCP 버퍼에 쌓였다가 회복 후 3ms 간격으로 몰려 들어왔다)

    끄려면 ENABLE_QUICK_EDIT_MODE 를 내리는 것만으로는 안 되고
    ENABLE_EXTENDED_FLAGS 를 같이 올려야 한다 - 이 플래그가 있어야
    SetConsoleMode 가 빠른편집/삽입 비트를 실제로 반영한다.

    콘솔이 아닌 곳(파일·파이프 리다이렉트, IDE 실행 등)에서는 GetConsoleMode
    가 실패하므로 조용히 아무것도 안 한다.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    STD_INPUT_HANDLE = -10
    ENABLE_QUICK_EDIT_MODE = 0x0040
    ENABLE_EXTENDED_FLAGS = 0x0080

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    # 핸들은 64비트다. restype 을 안 정하면 ctypes 가 32비트 int 로 잘라
    # 받는다(값이 작아 대개 표는 안 나지만 옳지 않다).
    k.GetStdHandle.restype = wintypes.HANDLE
    k.GetStdHandle.argtypes = [wintypes.DWORD]
    k.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]

    h = k.GetStdHandle(STD_INPUT_HANDLE)
    mode = wintypes.DWORD()
    if not k.GetConsoleMode(h, ctypes.byref(mode)):
        return None                      # 콘솔이 아니다
    new = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
    if not k.SetConsoleMode(h, new):
        return None
    return (k, h, mode.value)


def restore_console_mode(saved):
    """빠른 편집 설정을 원래대로.

    콘솔 모드는 프로세스가 아니라 '콘솔 창'의 속성이라 스크립트가 끝나도
    남는다. 되돌려 주지 않으면 그 창에서 마우스로 텍스트 선택을 못 하게
    된다 - 남의 창을 조용히 바꿔 놓는 셈이라 반드시 복구한다.
    """
    if not saved:
        return
    k, h, mode = saved
    k.SetConsoleMode(h, mode)


def local_ips():
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        return sorted({i[4][0] for i in infos})
    except OSError:
        return []


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="NILM WiFi 수신기 (프로토콜 v3) - 주기별 데이터를 CSV로 저장")
    ap.add_argument("--port", type=int, default=5000, help="TCP 포트 (기본 5000)")
    ap.add_argument("--csv", metavar="FILE",
                    help="CSV 파일명. 이름만 주면 data/ 안에 저장한다"
                         " (기본: data/nilm_<날짜시각>.csv, 있으면 이어쓰기)")
    ap.add_argument("--no-csv", action="store_true", help="CSV 저장 안 함")
    ap.add_argument("--quiet", action="store_true", help="콘솔 요약 끄기")
    args = ap.parse_args()

    # 콘솔 클릭 한 번에 수신이 멈추는 것을 막는다(위 함수 주석 참조).
    # 종료할 때 finally 에서 원래대로 되돌린다.
    console_mode = disable_console_quickedit()

    fcsv = writer = None
    csv_path = None
    if not args.no_csv:
        csv_path = resolve_csv_path(args.csv)
        fcsv, writer = open_csv(csv_path)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", args.port))
    except OSError as e:
        print(f"포트 {args.port} 를 열 수 없습니다: {e}")
        return 1
    srv.listen(1)
    # accept() 를 블로킹으로 두면 Windows에서 Ctrl+C 가 먹지 않는다. 파이썬은
    # 바이트코드 사이에서만 시그널을 처리하는데 블로킹 소켓 호출은 C 안에서
    # 멈춰 있기 때문이다. 접속 대기 중에 종료하면 최종 요약도 못 찍는다.
    # 1초마다 풀어 주면 그 틈에 KeyboardInterrupt 가 올라온다.
    srv.settimeout(1.0)

    ips = local_ips()
    print(f"listening on 0.0.0.0:{args.port}   (Ctrl+C 로 종료)")
    print(f"이 PC의 IPv4: {', '.join(ips) if ips else '(확인 실패)'}")
    if "192.168.137.1" not in ips:
        print("  주의: 192.168.137.1 이 없습니다. 모바일 핫스팟이 꺼져 있거나,"
              " 펌웨어의 NILM_HOST_IP 를 위 주소 중 하나로 바꿔야 합니다.")
    if csv_path:
        print(f"CSV: {csv_path}  (주기 1개 = 1행, 초당 60행)")
    else:
        print("CSV: 저장 안 함")

    frames = 0
    lost = 0
    stats = {"crc_err": 0, "resync": 0, "dup": 0,
             "ack_slow": 0, "ack_sent": 0, "ack_blocked": 0}
    cause_lost = {CAUSE_RECV: 0, CAUSE_BOARD: 0}
    # 재접속 경계를 넘어 사라진 프레임. 연결마다 seq_prev 를 새로 잡기 때문에
    # 예전에는 이게 통째로 안 세어졌고, 그 바람에 유실률이 실제보다 훨씬 작게
    # 보였다(실측 한 세션에서 보드는 52장을 버렸는데 여기 집계는 6장이었다).
    # 링크가 죽어 있는 동안 보드 큐가 넘쳐 버린 것이라 원인도 대응도 다르므로
    # 따로 센다.
    lost_gap = 0
    seq_global = None   # 연결을 넘어 이어지는 마지막 seq
    t_wall0 = time.time()

    try:
        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue                # 시그널 처리 틈 - 정상 경로
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[+] connected from {addr[0]}:{addr[1]}")
            seq_prev = None
            seq_base = None
            seq_seen = set()    # 중복 판정용 (펌웨어가 재전송한다)
            # 순서 뒤바뀜을 견디는 유실 집계용 (REORDER_MAX 주석 참조)
            seq_max = None      # 이 연결에서 본 가장 큰 seq
            lost_scan = None    # 여기까지는 유실 판정이 끝났다
            got_seq = set()     # 아직 판정 안 끝난 구간에서 받은 seq
            pending = {}        # seq -> (사건 순간의 카운터 증분, 벽시계 간격)
            c_frames = 0        # 이 연결에서만의 수신/유실
            c_lost = 0
            # 유실 원인 판정용: 직전 프레임 시점의 파서 카운터와 도착 시각
            stat_prev = dict(stats)
            t_prev = time.time()
            try:
                for payload in frame_stream(conn, stats, seq_seen, srv):
                    t_recv = time.time()
                    d = parse_payload(payload)
                    frames += 1
                    c_frames += 1

                    # 이 연결의 첫 프레임이면, 끊겨 있던 동안 사라진 것을 센다.
                    # seq 는 재접속해도 이어지므로(보드가 리셋되지 않는 한)
                    # 직전 연결의 마지막 seq 와 비교하면 공백이 그대로 드러난다.
                    # 뒤로 갔다면 보드가 리셋된 것이라 유실이 아니다.
                    if seq_prev is None and seq_global is not None:
                        if d["seq"] > seq_global + 1:
                            gap = d["seq"] - seq_global - 1
                            lost_gap += gap
                            print(f"[!] 끊겨 있는 동안 {gap}프레임 유실 "
                                  f"(seq {seq_global + 1}..{d['seq'] - 1}"
                                  f" = {gap * 0.5:.1f}초 분량)"
                                  f"  -> 보드 큐가 넘쳤습니다"
                                  f" (펌웨어 로그의 drop 과 맞춰 보세요)")
                        elif d["seq"] <= seq_global:
                            print("[!] seq 가 되돌아감 - 보드가 리셋된 것 같습니다.")

                    # --- 순서 뒤바뀜을 전제로 한 유실 집계 -----------------
                    #
                    # 펌웨어가 송신 윈도우를 쓰면서부터 프레임이 순서대로 오지
                    # 않는다. 새 프레임을 계속 내보내는 동안 확인 못 받은 옛
                    # 장을 골라 재전송하기 때문이다(선택적 재전송). 실측에서
                    #   37 -> 28 -> 38 -> 29 -> 39 -> 30
                    # 처럼 들어왔다. 28~30 은 앞서 CRC 오류로 죽었던 것이 뒤늦게
                    # 복구된 것이라 유실이 아니라 오히려 회수된 데이터다.
                    #
                    # 예전 집계는 seq 가 단조 증가한다고 보고 (a) 뒤로 간 것을
                    # "보드 리셋"으로 오인해 시간축을 리셋하고 (b) 앞뒤로 유령
                    # 공백을 셌다. 그래서 실제로 거의 안 잃었는데도 41% 로
                    # 보였다.
                    #
                    # 그래서 도착 순서로 판단하지 않는다. 받은 seq 를 모아 두고,
                    # "창을 완전히 벗어날 만큼 지나간" seq 만 유실로 확정한다.
                    s = d["seq"]
                    if seq_base is None:
                        seq_base = s
                        seq_max = s
                        lost_scan = s
                    elif s + REORDER_MAX < seq_max:
                        # 창 범위를 한참 벗어나 뒤로 갔다 = 진짜 보드 리셋
                        print("[!] seq 가 되돌아감 - 보드가 리셋된 것 같습니다."
                              " 시간축을 다시 잡습니다.")
                        seq_base = s
                        seq_max = s
                        lost_scan = s
                        got_seq.clear()
                    elif s > seq_max:
                        seq_max = s
                    got_seq.add(s)
                    pending.pop(s, None)     # 뒤늦게 왔다 - 후보에서 뺀다

                    # 도착 순서로 공백이 보이면 그 순간의 카운터를 떠 둔다.
                    # 유실 확정은 16초 뒤라, 그때 가서 델타를 재면 엉뚱한
                    # 시점의 값이 잡힌다. 원인 판정은 사건이 난 순간의
                    # CRC/재동기 증분으로 해야 의미가 있다.
                    if (seq_prev is not None) and (s > seq_prev + 1):
                        snap = {k: stats[k] - stat_prev[k] for k in stats}
                        dw = t_recv - t_prev
                        for q in range(seq_prev + 1, min(s, seq_prev + 1 + 64)):
                            if q not in got_seq:
                                pending.setdefault(q, (snap, dw))

                    # 이제 와서는 절대 안 올 seq 를 유실로 확정한다.
                    while lost_scan + REORDER_MAX < seq_max:
                        if lost_scan not in got_seq:
                            lost += 1
                            c_lost += 1
                            snap, dw = pending.pop(
                                lost_scan,
                                ({k: 0 for k in stats}, 0.0))
                            cause, why = classify_loss(snap, dw)
                            cause_lost[cause] += 1
                            print(f"[!] 프레임 유실 확정 (seq {lost_scan}) | "
                                  f"CRC +{snap['crc_err']} "
                                  f"재동기 +{snap['resync']} "
                                  f"중복 +{snap['dup']}  -> {why}")
                        else:
                            got_seq.discard(lost_scan)   # 다 쓴 것은 버린다
                        lost_scan += 1

                    seq_prev = s
                    seq_global = max(seq_global or s, s)
                    stat_prev = dict(stats)
                    t_prev = t_recv

                    if writer is not None:
                        write_frame(writer, d, t_recv, seq_base)
                        fcsv.flush()        # 중간에 죽어도 데이터는 남는다

                    if not args.quiet:
                        print(summary_line(d))
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                print(f"[-] 연결 끊김: {e}")
            finally:
                conn.close()
                # 연결이 끝났으니 아직 판정 안 끝난 구간을 마저 센다.
                # 이걸 안 하면 마지막 REORDER_MAX 개 구간의 유실이 통째로
                # 집계에서 빠져 유실률이 실제보다 작게 나온다.
                if lost_scan is not None and seq_max is not None:
                    tail = {CAUSE_RECV: 0, CAUSE_BOARD: 0}
                    while lost_scan <= seq_max:
                        if lost_scan not in got_seq:
                            snap, dw = pending.pop(
                                lost_scan, ({k: 0 for k in stats}, 0.0))
                            cause, _ = classify_loss(snap, dw)
                            cause_lost[cause] += 1
                            tail[cause] += 1
                        lost_scan += 1
                    n_tail = tail[CAUSE_RECV] + tail[CAUSE_BOARD]
                    if n_tail:
                        lost += n_tail
                        c_lost += n_tail
                        print(f"[!] 연결 종료 시점의 미판정 구간에서"
                              f" {n_tail}프레임 유실 확정"
                              f" (수신기 폐기(도착했으나 깨짐)"
                              f" {tail[CAUSE_RECV]},"
                              f" 보드가 안 보냄 {tail[CAUSE_BOARD]})")
                print(f"[-] disconnected ({loss_str(c_frames, c_lost)}), "
                      f"재접속 대기...")
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        restore_console_mode(console_mode)
        if fcsv is not None:
            fcsv.close()
        print_summary(frames, lost, stats, time.time() - t_wall0, csv_path,
                      cause_lost, lost_gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
