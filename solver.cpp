#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <set>

using namespace std;

const int INF = 1e9;

const int N = 13; // # values
const int K = 4;  // # colors
const int M = 3;  // # copies
const int S = 3;  // # min len

// n×k×f(M) multi-dimensional array which contains the maximum
// score that can be obtained given this state of the puzzle
vector<vector<int>> scores;
vector<vector<int>> scoresMemory;

vector<vector<int>> hand(N, vector<int>(K));
vector<vector<int>> board(N, vector<int>(K));;

vector<vector<int>> allRuns;
int maxHash;

// 4 * K 
int hashRun(vector<int> &run)
{
    int hash = 0;
    int pow = 1;
    sort(run.rbegin(), run.rend());
    for (int i = 0; i < run.size(); i++)
    {
        hash += pow * run[i];
        pow *= (S + 1);
    }
    return hash;
}

// 4 * K 
vector<int> unhashRun(int hash)
{
    vector<int> runs(M);
    for (int i = 0; i < runs.size(); i++)
    {
        runs[i] = hash % (S + 1);
        hash /= (S + 1);
    }
    return runs;
}

int combineHashes(vector<int> &hashes)
{
    int pow = 1;
    int result = 0;
    for (int hash : hashes)
    {
        result += pow * hash;
        pow *= maxHash;
    }
    return result;
}

vector<int> splitHashes(int totalHash)
{
    vector<int> hashes(K);
    for (int &hash : hashes)
    {
        hash = totalHash % maxHash;
        totalHash /= maxHash;
    }
    return hashes;
}

void init()
{
    maxHash = 1;
    for (int i = 0; i < M; i++)
    {
        maxHash *= (S + 1);
    }
    for (int hash = 0; hash < maxHash; hash++)
    {
        allRuns.push_back(unhashRun(hash));
    }
    int maxTotalHash = 1;
    for (int i = 0; i < K; i++)
    {
        maxTotalHash *= maxHash;
    }
    scores.resize(N, vector<int>(maxTotalHash, -INF));
    scoresMemory.resize(N, vector<int>(maxTotalHash, -INF));
}

int makeGroups(vector<int> &groups)
{
    int rest = 0;
    int j;
    for (j = 0; j < M; j++)
    {
        int groupSize = 0;
        vector<int> usedGroups(K);
        for (int i = 0; i < K; i++)
        {
            if (groups[i] - usedGroups[i] > 0)
            {
                groupSize++;
                usedGroups[i]++;
            }
        }
        if (groupSize + rest < S)
        {
            break;
        }
        rest += groupSize - S;
        for (int i = 0; i < K; i++)
        {
            groups[i] -= usedGroups[i];
        }
    }
    return j;
}

vector<pair<vector<int>, int>> makeRuns(vector<int> runsHashes, int value)
{
    vector<vector<int>> runs;
    for (int runHash : runsHashes)
    {
        runs.push_back(unhashRun(runHash));
    }
    vector<vector<array<int, 2>>> newRunsCombinations;
    for (int color = 0; color < K; color++)
    {
        set<array<int, 2>> newRuns;
        for (vector<int> &newRun : allRuns)
        {
            bool correct = true;
            int count = hand[value][color] + board[value][color];
            for (int i = 0; i < newRun.size() && count >= 0 && correct; i++)
            {
                if (newRun[i] == runs[color][i] + 1) // continue run
                {
                    count--;
                }
                else if (runs[color][i] == 3 && newRun[i] == 3) // continue run with 3 or more tiles
                {
                    count--;
                }
                else if (runs[color][i] == 3 && newRun[i] == 0) // finish run with 3 or more tiles
                {
                    // not use tiles
                }
                else if (runs[color][i] == 0 && newRun[i] == 0) // do nothing with empty run
                {
                    // not use tiles
                }
                else // else achive newRun from runs[color] is impossible
                {
                    correct = false;
                }
            }
            if (correct && count >= 0)
            {
                newRuns.insert({hashRun(newRun), count});
            }
        }
        
        if (newRuns.empty())
        {
            return {};
        }

        if (color == 0)
        {
            for (array<int, 2> e : newRuns)
            {
                newRunsCombinations.push_back(vector<array<int, 2>>(K));
                newRunsCombinations.back()[color] = e;
            }
            continue;
        }
        vector<vector<array<int, 2>>> newRunsCombinations_; // oooof
        for (vector<array<int, 2>> combination : newRunsCombinations)
        {
            for (array<int, 2> e : newRuns)
            {
                combination[color] = e;
                newRunsCombinations_.push_back(combination);
            }
        }
        newRunsCombinations = newRunsCombinations_; // oooof
    }
    vector<pair<vector<int>, int>> result; // newRunsHashes / scores
    for (vector<array<int, 2>> & combination : newRunsCombinations)
    {
        vector<int> newRunsHashes(K);
        vector<int> groups(K);
        for (int i = 0; i < K; i++)
        {
            newRunsHashes[i] = combination[i][0];
            groups[i] = combination[i][1];
        }

        makeGroups(groups);

        bool useAllBoard = true;
        int score = 0;
        for (int i = 0; i < K; i++)
        {
            int usedCnt = board[value][i] + hand[value][i] - groups[i];
            if (board[value][i] > usedCnt)
            {
                useAllBoard = false;
                break;
            }
            score += usedCnt;// * (value + 1);
        }
        if (useAllBoard)
        {
            result.push_back({newRunsHashes, score});
        }
    }
    return result;
}

bool checkRunsFinished(vector<int> &runsHashes)
{
    for (int runHash : runsHashes)
    {
        vector<int> run = unhashRun(runHash);
        for (int e : run)
        {
            if (e > 0 && e < S)
            {
                return false;
            }
        }
    }
    return true;
}

int maxScore(int value, vector<int> &runsHashes)
{
    if (value >= N)
    {
        if (checkRunsFinished(runsHashes))
        {
            return 0;
        }
        else
        {
            return -INF;
        }
    } 

    int totalRunsHash = combineHashes(runsHashes);
    if (scores[value][totalRunsHash] > -INF)
    {
        return scores[value][totalRunsHash];
    }

    vector<pair<vector<int>, int>> newRunsHashes = makeRuns(runsHashes, value);
    for (int i = 0; i < newRunsHashes.size(); i++)
    {
        int curScore = newRunsHashes[i].second;
        int nextScore = maxScore(value+1, newRunsHashes[i].first);
        if (nextScore > -INF)
        {
            if (scores[value][totalRunsHash] < curScore+nextScore)
            {
                scores[value][totalRunsHash] = curScore+nextScore;
                scoresMemory[value][totalRunsHash] = combineHashes(newRunsHashes[i].first);
            }
        }
    }
    return scores[value][totalRunsHash];
}

vector<vector<pair<int, int>>> restore()
{
    vector<vector<pair<int, int>>> rows;

    vector<vector<int>> runs(K, vector<int>(M));
    // vector<int> groups[M];
    int totalRunsHash = 0;
    for (int value = 0; value < N; value++)
    {
        vector<int> groups(K);
        totalRunsHash = scoresMemory[value][totalRunsHash];
        vector<int> runsHashes = splitHashes(totalRunsHash);
        for (int i = 0; i < K; i++)
        {
            vector<int> run = unhashRun(runsHashes[i]);
            
            int cnt = 0;
            for (int r : run)
            {
                if (r > 0)
                {
                    cnt++;
                }
            }
            sort(runs[i].begin(), runs[i].end(), [](int a, int b){return (a < b) && (a > 0);});
            for (int j = 0; j < cnt; j++)
            {
                runs[i][j]++;
                if (value == N-1)
                {
                    int len_run = runs[i][j];
                    rows.push_back(vector<pair<int, int>>(len_run));
                    for (int k = 0; k < len_run; k++)
                    {
                        rows[rows.size() - 1][k] = {value - len_run + 1 + k, i};
                    }
                    runs[i][j] = 0;
                }
            }
            for (int j = cnt; j < M && runs[i][j] > 0; j++)
            {
                int len_run = runs[i][j];
                rows.push_back(vector<pair<int, int>>(len_run));
                for (int k = 0; k < len_run; k++)
                {
                    rows[rows.size() - 1][k] = {value - len_run + k, i};
                }
                runs[i][j] = 0;
            }

            groups[i] = hand[value][i] + board[value][i] - cnt; 
        }
        vector<int> groups_copy = groups;
        int n_groups = makeGroups(groups_copy);
        for (int i = 0; i < n_groups; i++)
            rows.push_back({});
        for (int i = 0; i < K; i++)
        {
            sort(rows.end()-n_groups, rows.end(), [](vector<pair<int, int>> &a, vector<pair<int, int>> &b){return (a.size() > b.size()) || (a.size() > 0);});
            for (int j = 0; j < groups[i] && j < n_groups; j++)
            {
                rows[rows.size() - 1 - j].push_back({ value, i });
            }
        }
    }

    return rows;
}

int main()
{
    init();
    
    // for (int i = 0; i < N; i++)
    // {
    //     board[i][0] = 1;
    // }
    // board[12][0] = 1;
    // board[11][0] = 1;
    // board[10][0] = 1;
    // hand[1][2] = 1;
    // hand[1][3] = 1;
    // hand[1][0] = 2;
    // hand[2][0] = 1;
    // hand[3][0] = 1;
    // hand[8][0] = 1;
    hand[9][0] = 1;
    hand[9][1] = 2;
    hand[9][2] = 1;
    hand[9][3] = 1;
    // hand[10][0] = 1;

    vector<int> runsHashes(K);
    int score = maxScore(0, runsHashes);
    cout << score << "\n";
    if (score > -INF)
    {
        cout << "Restore start\n";
        vector<vector<pair<int, int>>> rows = restore();
        cout << "n rows: " << rows.size() << "\n";
        for (auto &row : rows)
        {
            for (auto tile : row)
            {
                cout << tile.first << "-" << tile.second << " ";
            }
            cout << "\n";
        }
    }

    return 0;
}