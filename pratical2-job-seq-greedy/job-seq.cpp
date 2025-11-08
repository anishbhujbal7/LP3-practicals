#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Job {
    char id;      // Job ID (like A, B, C...)
    int deadline; // Deadline of job
    int profit;   // Profit if job is completed before deadline
};

// Comparison function to sort jobs by profit (descending)
bool compare(Job a, Job b) {
    return a.profit > b.profit;
}

void jobSequencing(vector<Job> &jobs) {
    // Step 1: Sort jobs by decreasing profit
    sort(jobs.begin(), jobs.end(), compare);

    int n = jobs.size();
    int maxDeadline = 0;

    // Find maximum deadline
    for (auto &job : jobs)
        if (job.deadline > maxDeadline)
            maxDeadline = job.deadline;

    // Step 2: Create a schedule array to keep track of free time slots
    vector<char> slot(maxDeadline + 1, '-'); // '-' means empty
    int totalProfit = 0;

    // Step 3: Assign jobs to slots
    for (int i = 0; i < n; i++) {
        // Try to find a free slot before the job's deadline
        for (int j = jobs[i].deadline; j > 0; j--) {
            if (slot[j] == '-') {
                slot[j] = jobs[i].id; // Assign job
                totalProfit += jobs[i].profit;
                break;
            }
        }
    }

    // Step 4: Print result
    cout << "\nJob Sequence: ";
    for (int i = 1; i <= maxDeadline; i++) {
        if (slot[i] != '-')
            cout << slot[i] << " ";
    }

    cout << "\nTotal Profit: " << totalProfit << endl;
}

int main() {
    int n;
    cout << "Enter number of jobs: ";
    cin >> n;

    vector<Job> jobs(n);
    cout << "Enter Job ID, Deadline, and Profit:\n";
    for (int i = 0; i < n; i++) {
        cin >> jobs[i].id >> jobs[i].deadline >> jobs[i].profit;
    }

    jobSequencing(jobs);

    return 0;
}


// Enter number of jobs: 5
// Enter Job ID, Deadline, and Profit:
// A 2 100
// B 1 19
// C 2 27
// D 1 25
// E 3 15

// Step	Operation	Complexity
// Sorting	O(n log n)	Sorting by profit
// Scheduling	O(n * maxDeadline)	Checking available slots
// Total	O(n log n + n × maxDeadline)	Efficient for small deadlines

