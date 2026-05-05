"""Build a small local index of ISCO-08 occupations + role labels.

For v1 we ship a hand-curated CSV of common occupations spanning the major groups.
Running this script seeds data/esco/occupations.csv if missing. Stretch: pull from
ESCO API at https://ec.europa.eu/esco/api/resource/occupation
"""
from pathlib import Path
import csv

OUT = Path("data/esco/occupations.csv")

# Minimal seed: 4-digit ISCO codes covering common occupations across major groups.
# Format: isco_code,label_en,label_cs,keywords_en,keywords_cs
SEED = [
    ("1120", "Managing director", "Generální ředitel", "ceo, managing director, executive", "ředitel, ceo"),
    ("1330", "ICT services manager", "Manažer IT služeb", "head of engineering, vp engineering, cto", "vedoucí it, manažer it"),
    ("2221", "Nursing professional", "Všeobecná zdravotní sestra", "registered nurse, nurse", "zdravotní sestra, sestra"),
    ("2310", "University and higher education teacher", "Vysokoškolský učitel", "professor, lecturer", "profesor, docent"),
    ("2330", "Secondary education teacher", "Učitel střední školy", "high school teacher", "středoškolský učitel"),
    ("2411", "Accountant", "Účetní", "accountant, controller", "účetní"),
    ("2421", "Management consultant", "Manažerský poradce", "management consultant, strategy consultant", "konzultant, poradce"),
    ("2434", "ICT sales professional", "Obchodník v IT", "sales engineer, account executive, ict sales", "obchodník it, sales it"),
    ("2511", "Systems analyst", "Systémový analytik", "systems analyst, business analyst", "systémový analytik, business analytik"),
    ("2512", "Software developer", "Vývojář software", "software developer, software engineer, backend, full stack, frontend", "programátor, vývojář, software engineer"),
    ("2513", "Web and multimedia developer", "Webový vývojář", "web developer, frontend developer", "webový vývojář, frontend"),
    ("2514", "Applications programmer", "Aplikační programátor", "applications programmer, mobile developer", "aplikační programátor"),
    ("2519", "Software and applications developer NEC", "Ostatní vývojáři SW", "QA engineer, test engineer, automation, sdet", "QA inženýr, tester, automatizace testování"),
    ("2521", "Database designer and administrator", "DBA", "database administrator, dba, data engineer", "dba, datový inženýr"),
    ("2522", "Systems administrator", "Systémový administrátor", "systems administrator, sysadmin, devops, sre", "sysadmin, devops, sre"),
    ("2523", "Computer network professional", "Síťový specialista", "network engineer", "síťový inženýr"),
    ("2529", "Database and network professionals NEC", "Ostatní DB/sítě", "data scientist, ml engineer, ai engineer, machine learning", "data scientist, ml inženýr, AI inženýr"),
    ("2611", "Lawyer", "Právník", "lawyer, attorney", "právník, advokát"),
    ("2632", "Sociologist, anthropologist", "Sociolog", "sociologist, anthropologist", "sociolog"),
    ("2641", "Author and related writer", "Spisovatel", "author, copywriter, content writer", "autor, copywriter"),
    ("2651", "Visual artist", "Výtvarný umělec", "graphic designer, visual artist", "grafik, designer"),
    ("3221", "Nursing associate professional", "Praktická sestra", "nursing assistant, practical nurse", "praktická sestra"),
    ("3322", "Commercial sales representative", "Obchodní zástupce", "sales representative, account manager", "obchodní zástupce, account manager"),
    ("4110", "General office clerk", "Administrativní pracovník", "office administrator, secretary", "administrativní pracovník, sekretářka"),
    ("5120", "Cook", "Kuchař", "cook, chef", "kuchař"),
    ("5223", "Shop sales assistant", "Prodavač", "shop assistant, retail", "prodavač"),
    ("7115", "Carpenter", "Tesař", "carpenter", "tesař"),
    ("8322", "Car driver", "Řidič osobních aut", "driver, taxi driver", "řidič"),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"{OUT} already exists; not overwriting. Delete to regenerate.")
        return
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["isco_code", "label_en", "label_cs", "keywords_en", "keywords_cs"])
        for row in SEED:
            w.writerow(row)
    print(f"Wrote {len(SEED)} occupations → {OUT}")


if __name__ == "__main__":
    main()
