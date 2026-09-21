"""Fetch bibliographic records from DOI registration agencies and arXiv."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

DOIS={
 'deepjscc':'10.1109/TCCN.2019.2919300',
 'swinjscc':'10.1109/TCCN.2024.3424842',
 'deepjscc_f':'10.1109/JSAIT.2020.2987203',
 'fineharq':'10.1109/TWC.2025.3532501',
 'cddm':'10.1109/TWC.2024.3379244',
 'ntscc':'10.1109/JSAC.2022.3180802',
 'lpips':'10.1109/CVPR.2018.00068',
 'div2k':'10.1109/CVPRW.2017.150',
 'msssim':'10.1109/ACSSC.2003.1292216',
}
ARXIV={'flowmatching':'2210.02747','flowsem':'2608.21651','rcbfm':'2607.24876','genharq':'2603.15068'}


def fetch(item):
    key,identifier,is_arxiv=item
    url=f'https://api.datacite.org/dois/10.48550/arxiv.{identifier}' if is_arxiv else f'https://api.crossref.org/works/{identifier}/transform/application/x-bibtex'
    req=urllib.request.Request(url,headers={'User-Agent':'FlowHARQ-research-bibliography-audit/1.0','Accept':'*/*' if is_arxiv else 'application/x-bibtex'})
    try:
        text=urllib.request.urlopen(req,timeout=45).read().decode('utf8').strip()
        metadata=None
        if is_arxiv:
            metadata=json.loads(text)['data']['attributes']
            authors=' and '.join(person['name'] for person in metadata['creators'])
            title=metadata['titles'][0]['title']
            text='@article{'+key+',\n author={'+authors+'},\n title={{'+title+'}},\n journal={arXiv preprint arXiv:'+identifier+'},\n year={'+str(metadata['publicationYear'])+'},\n doi={'+metadata['doi']+'}\n}'
        if not text.startswith('@'):
            raise ValueError('Response was not BibTeX')
        bib=re.sub(r'(@\w+\s*\{)[^,]+,',lambda m:m.group(1)+key+',',text,count=1)
        return key,dict(source=url,identifier=identifier,status='verified_provider_record',sha256=hashlib.sha256(text.encode()).hexdigest(),raw_bibtex=text,provider_metadata=metadata),bib
    except Exception as error:
        return key,dict(source=url,identifier=identifier,status='fetch_failed',error=str(error)),None


def main():
    items=[(key,value,False) for key,value in DOIS.items()]+[(key,value,True) for key,value in ARXIV.items()]
    destination=Path('experiments/spring_final_20260921')
    prior=json.loads((destination/'citation_records.json').read_text()) if (destination/'citation_records.json').exists() else {}
    records=[]
    import time
    for item in items:
        key=item[0]
        if prior.get(key,{}).get('status')=='verified_provider_record':
            raw=prior[key]['raw_bibtex']
            records.append((key,prior[key],re.sub(r'(@\w+\s*\{)[^,]+,',lambda m:m.group(1)+key+',',raw,count=1)))
        else:
            records.append(fetch(item))
            time.sleep(2)
    (destination/'citation_records.json').write_text(json.dumps({key:meta for key,meta,_ in records},indent=2)+'\n')
    failed=[key for key,_,bib in records if bib is None]
    if failed:
        raise RuntimeError('Unverified records: '+','.join(failed))
    normalized=[]
    for key,_,bib in records:
        bib=re.sub(r'</?i>','',html.unescape(bib)).replace('&',r'\&').replace('–','--')
        bib=re.sub(r'\s+', ' ', bib)
        bib=re.sub(r'\bmonth\s*=\s*([A-Za-z]+)',lambda m:'month='+m.group(1).lower()[:3],bib)
        bib=bib.replace('DeepJSCC- f :','DeepJSCC-f:').replace('Thrity-Seventh','Thirty-Seventh')
        for acronym in ('DeepJSCC','SwinJSCC','CDDM','HARQ','NTIRE'):
            # Protect title acronyms without altering DOI or citation keys.
            bib=re.sub(r'(title=\{)(.*?)(\},)',lambda m:m.group(1)+m.group(2).replace(acronym,'{'+acronym+'}')+m.group(3),bib,count=1)
        if key=='msssim' and not re.search(r'\byear\s*=',bib):
            # Proceedings date is in the Crossref-provided conference title.
            assert '2003' in bib
            bib=bib.rstrip().rstrip('}')+', year={2003}}'
        normalized.append(bib)
    Path('paper/spring_references.bib').write_text('\n\n'.join(normalized)+'\n')
    print('Provider-verified references:',len(records))


if __name__=='__main__':
    main()
