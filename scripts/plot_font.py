# scripts/plot_font.py
from matplotlib import font_manager, rcParams
from pathlib import Path
from typing import Optional

def set_kr_font() -> Optional[str]:
    """
    1) assets/fonts의 TTF/OTF를 등록(addfont)
    2) 선호 순서대로 설치된 폰트를 기본 폰트로 지정
    3) 사용한 폰트명을 반환(없으면 None)
    """
    # scripts/plot_font.py 기준으로 리포 루트(../)
    root = Path(__file__).resolve().parents[1]
    font_dir = root / "assets" / "fonts"

    # 1) 번들 폰트 등록(있으면)
    if font_dir.is_dir():
        for f in list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf")):
            try:
                font_manager.fontManager.addfont(str(f))
            except Exception:
                pass  # 일부 깨진 파일이 있어도 전체 중단하지 않음

    # 2) 후보 우선순위(동봉 → 리눅스 → 윈도우 → 맥)
    preferred = [
        "NanumGothic",
        "Nanum Gothic",
        "Noto Sans CJK KR",
        "Malgun Gothic",
        "Apple SD Gothic Neo",
    ]

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return name
    return None

def get_kr_font_path(preferred=None) -> Optional[str]:
    """
    워드클라우드용 font_path 절대경로 반환.
    우선순위:
      1) assets/fonts 내 TTF/OTF(파일명/내부명 패턴 매칭)
      2) 시스템 폰트 캐시에서 선호 이름 매칭
    """
    from matplotlib import font_manager

    root = Path(__file__).resolve().parents[1]
    font_dir = root / "assets" / "fonts"
    candidates = preferred or [
        "NanumGothic",
        "Nanum Gothic",
        "Noto Sans CJK KR",
        "Malgun Gothic",
        "Apple SD Gothic Neo",
    ]

    # 1) 번들 폰트 파일 우선
    if font_dir.is_dir():
        files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
        if files:
            # 이름 힌트 기준으로 우선 선택(없으면 첫 파일)
            for name in candidates:
                for f in files:
                    low = f.name.lower()
                    if any(k in low for k in ["nanum", "noto", "gothic", "apple", "kr"]):
                        return str(f)
            return str(files[0])

    # 2) 시스템 폰트 캐시에서 이름 매칭
    for ft in font_manager.fontManager.ttflist:
        if ft.name in candidates:
            return ft.fname

    # 3) 최후의 보루: 윈도우 기본 경로 후보(환경에 따라 다를 수 있음)
    fallback_paths = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
    ]
    for p in fallback_paths:
        if Path(p).exists():
            return p

    return None