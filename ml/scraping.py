from autoscraper import AutoScraper

url = "https://github.com/TaufiqSameer?tab=repositories"

wanted = ["AIING"]

scraper = AutoScraper()
result = scraper.build(url,wanted_list=wanted)
print(result)
scraper.set_rule_aliases({
    "rule_stars" : "stars"
})
scraper.keep_rules(["rule_stars"])
print(result["rule_stars"]);
