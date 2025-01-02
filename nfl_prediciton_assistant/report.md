Database containing all the embedded data from the CSV an JSON file present in config 


```sql
-- Sample Schema for NFL Data (adjust based on your actual data)

CREATE TABLE Teams (
  team_id INT PRIMARY KEY,
  team_name VARCHAR(255),
  conference VARCHAR(255),
  division VARCHAR(255)
);

CREATE TABLE Players (
  player_id INT PRIMARY KEY,
  player_name VARCHAR(255),
  position VARCHAR(255),
  team_id INT,
  FOREIGN KEY (team_id) REFERENCES Teams(team_id)
);

CREATE TABLE Games (
  game_id INT PRIMARY KEY,
  home_team_id INT,
  away_team_id INT,
  score_home INT,
  score_away INT,
  date DATE
);


```