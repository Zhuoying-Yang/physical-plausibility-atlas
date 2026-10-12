#!/usr/bin/env python3
"""Find evaluated original clips under Xiaoshan, compress for GitHub Pages, and make a manifest."""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

VIDEO_EXTS={'.mp4','.mov','.webm','.m4v'}
MAX_BYTES=780*1024*1024  # Leave some headroom under GitHub Pages' 1GB limit
MAX_CLIP=80*1024*1024


def detect_id(path,root):
    try: rel=path.relative_to(root).as_posix().lower()
    except ValueError:return None
    parts=rel.split('/'); name=path.stem.lower()
    m=re.search(r'cosmos3_seed(10[123])_(\d{4})',rel)
    if m:return f'Cosmos3_seed{m.group(1)}_{m.group(2)}'
    m=re.search(r'cosmos2(?:\.5|5)_(\d{4})',rel)
    if m:return 'Cosmos2.5_'+m.group(1)
    m=re.search(r'(?:^|/)lvp_(\d{4})',rel)
    if m:return 'LVP_'+m.group(1)
    nums=[x for x in parts[:-1] if re.fullmatch(r'\d{4}',x)]
    m=re.match(r'^(\d{4})(?:$|[_-])',name)
    num=nums[-1] if nums else (m.group(1) if m else None)
    if num is None:return None
    if 'cosmos3' in rel:
        s=next((x for x in parts if re.fullmatch('seed10[123]',x)),None)
        return f'Cosmos3_{s}_{num}' if s else None
    if 'cosmos2.5' in rel or 'cosmos25' in rel:return f'Cosmos2.5_{num}'
    if 'lvp' in rel or 'robowm_lvp' in rel:return f'LVP_{num}'
    return None


def priority(p):
    s=p.as_posix().lower(); fn=p.name.lower()
    derived=bool(re.search(r'overlay|mask|depth|flow|track|montage|annot|segmen|crop|thumbnail|roi|side.by.side|comparison|debug|visualiz',s))
    orig=bool(re.fullmatch(r'(?:\d{4}|video|original|raw|rgb|input|source|lvp_\d{4}|cosmos2\.5_\d{4}|cosmos3_seed10[123]_\d{4})\.(?:mp4|mov|webm|m4v)',fn))
    return (derived,not orig,p.suffix.lower()!='.mp4',len(s))


def scan(root,ids):
    selected={}
    bases=[root/'Cosmos3',root/'Cosmos2.5'/'Videos',root/'LVP'/'robowm_lvp_records'/'robowm_lvp_68_numbered'/'videos']
    print('Scanning source directories:',flush=True)
    for base in bases:
        print('  '+str(base)+(' ✓' if base.is_dir() else ' [not found]'),flush=True)
        if not base.is_dir():continue
        for parent,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if not d.startswith('.') and d not in ('__MACOSX','node_modules','.git')]
            for filename in files:
                p=Path(parent)/filename
                if p.suffix.lower() not in VIDEO_EXTS:continue
                id=detect_id(p,root)
                if id not in ids:continue
                if id not in selected or priority(p)<priority(selected[id]):selected[id]=p
    return selected


def encode(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_name(dst.stem+'.tmp.mp4')
    tmp.unlink(missing_ok=True)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-i',str(src),
         '-map','0:v:0','-an',
         '-vf','scale=960:720:force_original_aspect_ratio=decrease:force_divisible_by=2',
         '-r','24','-c:v','libx264','-preset','veryfast','-b:v','700k','-maxrate','900k',
         '-bufsize','1800k','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    p=subprocess.run(cmd,text=True,capture_output=True)
    if p.returncode or not tmp.exists() or tmp.stat().st_size==0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'Could not encode {src}: {(p.stderr or "unknown error")[-500:]}')
    if tmp.stat().st_size>MAX_CLIP:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'Clip exceeds {MAX_CLIP//1024//1024} MiB: {src}')
    tmp.replace(dst)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',type=Path,default=Path.home()/'Downloads/Harvard/Xiaoshan')
    ap.add_argument('--site-root',type=Path,required=True)
    ap.add_argument('--case-ids',type=Path,default=Path(__file__).with_name('case_ids.json'))
    a=ap.parse_args()
    root=a.source_root.expanduser().resolve()
    ids=set(json.loads(a.case_ids.read_text()))
    if not root.is_dir():sys.exit(f'ERROR: No Xiaoshan source directory at {root}')
    if not shutil.which('ffmpeg'):sys.exit('ERROR: ffmpeg missing. Install with: brew install ffmpeg')
    out=a.site_root.resolve();vd=out/'videos';vd.mkdir(parents=True,exist_ok=True)
    found=scan(root,ids)
    if not found:sys.exit('ERROR: No matching source MP4 files. Check path or filenames.')
    print(f'Found originals for {len(found)}/{len(ids)} evaluated cases. Encoding MP4 copies (audio removed).',flush=True)
    manifest={}
    errors=[]
    for i,id in enumerate(sorted(found),1):
        src=found[id]
        dst=vd/(id+'.mp4')
        if not dst.exists():
            try:encode(src,dst)
            except Exception as e:errors.append((id,str(e)));print('FAILED',id,str(e),flush=True);continue
        sz=dst.stat().st_size
        manifest[id]={'url':'videos/'+id+'.mp4','bytes':sz}
        print(f'  [{i:03d}/{len(found):03d}] {id} {sz/1024/1024:.2f} MiB',flush=True)
    if errors:
        print(f'ERROR: {len(errors)} videos could not be encoded. Nothing will be pushed.',flush=True)
        return 1
    total=sum(x['bytes'] for x in manifest.values())
    if total>MAX_BYTES:
        print(f'ERROR: Video size {total/1024/1024:.1f} MiB exceeds safe budget of {MAX_BYTES/1024/1024:.0f} MiB. Nothing will be pushed.',flush=True)
        return 1
    result={'version':1,'published_count':len(manifest),'evaluated_count':len(ids),'total_video_mib':round(total/1024/1024,2),'videos':manifest}
    (vd/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f'READY: {len(manifest)} MP4 files / {total/1024/1024:.1f} MiB; missing source videos: {len(ids)-len(manifest)}',flush=True)
    if len(manifest)<len(ids):
        missing=sorted(ids-set(manifest))
        (out/'missing_videos.txt').write_text('\n'.join(missing)+'\n')
        print('Missing IDs recorded in unified-atlas/missing_videos.txt',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
