use std::cell::RefCell;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub struct Solution {}

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

impl Solution {
    // 1: https://leetcode-cn.com/problems/two-sum/
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut m = HashMap::new();
        for (i, num) in nums.iter().enumerate() {
            if let Some(j) = m.get(&(target - num)) {
                if *j != i {
                    return vec![i as i32, *j as i32];
                }
            }
            m.insert(num, i);
        }
        panic!("not found");
    }
    // 26: https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        if nums.len() == 0 {
            return 0;
        }
        let mut i = 0;
        let mut j = 0;
        while j < nums.len() {
            if nums[i] != nums[j] {
                i = i + 1;
                nums[i] = nums[j];
            }
            j = j + 1;
        }
        (i + 1) as i32
    }
    // 27: https://leetcode-cn.com/problems/remove-element/
    pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
        let mut i = 0;
        let mut j = 0;
        while j < nums.len() {
            if val != nums[j] {
                nums[i] = nums[j];
                i = i + 1;
            }
            j = j + 1;
        }
        i as i32
    }
    // 35: https://leetcode-cn.com/problems/search-insert-position/
    pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
        let (mut i, mut j) = (0, nums.len());
        while i < j {
            let mid = i + (j - i) / 2;
            let cur = nums[mid];
            if cur == target {
                return mid as i32;
            } else if cur < target {
                i = mid + 1;
            } else {
                j = mid;
            }
        }
        return i as i32;
    }
    // 53: https://leetcode-cn.com/problems/maximum-subarray/
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut sum = nums[0];
        let mut res = sum;
        for i in 1..nums.len() {
            if sum < 0 {
                sum = 0;
            }
            sum = sum + nums[i];
            res = cmp::max(sum, res);
        }
        return res;
    }
    // 66: https://leetcode-cn.com/problems/plus-one/
    pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
        let mut digits = digits;
        let mut carry = true;
        for digit in digits.iter_mut().rev() {
            if carry {
                *digit = (*digit + 1) % 10;
                carry = *digit == 0 && carry;
            } else {
                break;
            }
        }
        if carry {
            digits.insert(0, 1);
        }
        digits
    }
    // 88: https://leetcode-cn.com/problems/merge-sorted-array/
    pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
        let mut e1 = m - 1;
        let mut e2 = n - 1;
        for curr in (0..nums1.len()).rev() {
            if e2 < 0 || e1 >= 0 && nums1[e1 as usize] > nums2[e2 as usize] {
                nums1[curr] = nums1[e1 as usize];
                e1 = e1 - 1;
            } else {
                nums1[curr] = nums2[e2 as usize];
                e2 = e2 - 1;
            }
        }
    }
    // 108: https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/
    pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        fn split(nums: &Vec<i32>, s: usize, e: usize) -> Option<Rc<RefCell<TreeNode>>> {
            if e - s == 0 {
                return None;
            } else if e - s == 1 {
                return Some(Rc::new(RefCell::new(TreeNode::new(nums[s]))));
            }
            let m = s + (e - s) / 2;
            let root = Rc::new(RefCell::new(TreeNode::new(nums[m])));
            root.borrow_mut().left = split(nums, s, m);
            root.borrow_mut().right = split(nums, m + 1, e);
            return Some(root);
        }
        return split(&nums, 0, nums.len());
    }
    // 118: https://leetcode-cn.com/problems/pascals-triangle/
    pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
        let mut res = vec![vec![1]];
        for i in 1..num_rows as usize {
            let mut line = vec![0; i + 1];
            for j in 1..i {
                line[j] = res[i - 1][j] + res[i - 1][j - 1];
            }
            line[0] = 1;
            line[i] = 1;
            res.push(line);
        }
        res
    }
    // 119: https://leetcode-cn.com/problems/pascals-triangle-ii/
    pub fn get_row(row_index: i32) -> Vec<i32> {
        let mut res = Vec::with_capacity(row_index as usize + 1);
        res.push(1);
        (0..row_index).for_each(|i| {
            res.push((*res.last().unwrap() as i64 * (row_index - i) as i64 / (i + 1) as i64) as i32)
        });
        res
    }
    // 121: https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        if prices.len() < 2 {
            return 0;
        }
        let mut res = 0;
        let mut max = prices.last().unwrap();
        for i in (0..prices.len() - 1).rev() {
            res = cmp::max(res, max - prices[i]);
            max = cmp::max(max, &prices[i]);
        }
        res
    }
    // 136: https://leetcode-cn.com/problems/single-number/
    pub fn single_number(nums: Vec<i32>) -> i32 {
        nums.iter().fold(0, |acc, x| acc ^ *x)
    }
    // 169: https://leetcode-cn.com/problems/majority-element/
    pub fn majority_element(nums: Vec<i32>) -> i32 {
        let mut res = nums[0];
        let mut cnt = 1;
        for i in 1..nums.len() {
            if res == nums[i] {
                cnt += 1;
            } else {
                cnt = cnt - 1;
                if cnt == 0 {
                    res = nums[i];
                    cnt = 1;
                }
            }
        }
        res
    }
    // 217: https://leetcode-cn.com/problems/contains-duplicate/
    pub fn contains_duplicate(nums: Vec<i32>) -> bool {
        let set: HashSet<i32> = nums.clone().into_iter().collect();
        set.len() != nums.len()
    }
    // 219: https://leetcode-cn.com/problems/contains-duplicate-ii/
    // slide-window suits here. although the algorithm seems more delicate but the implementation
    // is more complicated.
    pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
        let mut m = HashMap::new();
        for i in 0..nums.len() {
            let num = nums[i];
            if let Some(j) = m.insert(num, i) {
                if i - j <= k as usize {
                    return true;
                }
            }
        }
        false
    }
    // 228: https://leetcode-cn.com/problems/summary-ranges/
    pub fn summary_ranges(nums: Vec<i32>) -> Vec<String> {
        let mut res = Vec::new();
        if nums.len() == 0 {
            return res;
        } else if nums.len() == 1 {
            return vec![format!("{}", nums[0])];
        }
        let mut s = 0;
        for i in 1..=nums.len() {
            if i == nums.len() || nums[i] - nums[i - 1] != 1 {
                if i - 1 - s >= 1 {
                    res.push(format!("{}->{}", nums[s], nums[i - 1]))
                } else {
                    res.push(format!("{}", nums[s]))
                }
                s = i;
            }
        }
        res
    }
    // 268: https://leetcode-cn.com/problems/missing-number/
    pub fn missing_number(nums: Vec<i32>) -> i32 {
        nums.iter()
            .fold(nums.len() * (nums.len() + 1) / 2, |acc, x| {
                acc - *x as usize
            }) as i32
    }
    // 448: https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/
    pub fn find_disappeared_numbers(nums: Vec<i32>) -> Vec<i32> {
        let mut nums = nums;
        for i in 0..nums.len() {
            let num = (nums[i].abs()) as usize - 1;
            nums[num] = -nums[num].abs();
        }
        nums.iter()
            .enumerate()
            .fold(Vec::new(), |mut res, (idx, &num)| {
                if num > 0 {
                    res.push(idx as i32 + 1);
                }
                res
            })
    }
    // 283: https://leetcode-cn.com/problems/move-zeroes/
    pub fn move_zeroes(nums: &mut Vec<i32>) {
        let mut s = 0;
        for i in 0..nums.len() {
            if nums[i] != 0 {
                nums[s] = nums[i];
                s += 1;
            }
        }
        for i in s..nums.len() {
            nums[i] = 0;
        }
    }
}

//303: https://leetcode-cn.com/problems/range-sum-query-immutable/
#[allow(dead_code)]
struct NumArray {
    sums: Vec<i32>,
}

impl NumArray {
    #[allow(dead_code)]
    fn new(nums: Vec<i32>) -> Self {
        let mut sum = 0;
        Self {
            sums: nums
                .iter()
                .map(|num| {
                    sum = *num + sum;
                    sum
                })
                .collect(),
        }
    }

    #[allow(dead_code)]
    fn sum_range(&self, left: i32, right: i32) -> i32 {
        let sum = if left == 0 {
            0
        } else {
            self.sums[left as usize - 1]
        };
        self.sums[right as usize] - sum
    }
}

impl Solution {
    // 349: https://leetcode-cn.com/problems/intersection-of-two-arrays/
    pub fn intersection(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        nums1
            .into_iter()
            .collect::<HashSet<i32>>()
            .intersection(&nums2.into_iter().collect::<HashSet<i32>>())
            .map(|&num| num)
            .collect()
    }
    // 350: https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/
    pub fn intersect(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        fn frequence(nums: &Vec<i32>) -> HashMap<i32, usize> {
            nums.iter().fold(HashMap::new(), |mut m, num| {
                *m.entry(*num).or_insert(0) += 1;
                m
            })
        }
        let m1 = frequence(&nums1);
        let m2 = frequence(&nums2);
        m1.iter().fold(Vec::new(), |mut res, (&num, &freq)| {
            res.append(&mut vec![num; cmp::min(freq, *m2.get(&num).unwrap_or(&0))]);
            res
        })
    }
    // 414: https://leetcode-cn.com/problems/third-maximum-number/
    pub fn third_max(nums: Vec<i32>) -> i32 {
        let mut seq = Vec::with_capacity(4);
        for i in 0..nums.len() {
            if !seq.contains(&nums[i]) {
                seq.push(nums[i]);
                seq.sort_by(|a, b| b.cmp(a));
                if seq.len() > 3 {
                    seq.pop();
                }
            }
        }
        if seq.len() < 3 {
            *seq.first().unwrap()
        } else {
            *seq.last().unwrap()
        }
    }
    // 453: https://leetcode-cn.com/problems/minimum-moves-to-equal-array-elements-ii/
    pub fn min_moves_to_equal_array_elements_ii(nums: Vec<i32>) -> i32 {
        nums.iter().sum::<i32>() - nums.iter().min().unwrap() * (nums.len() as i32)
    }
    // 455: https://leetcode-cn.com/problems/assign-cookies/
    pub fn find_content_children(g: Vec<i32>, s: Vec<i32>) -> i32 {
        let mut g = g;
        let mut s = s;
        s.sort();
        g.sort();
        let mut res = 0;
        let mut i = 0;
        for cookie in s {
            if i >= g.len() {
                break;
            }
            if g[i] <= cookie {
                res += 1;
                i += 1;
            }
        }
        res
    }
    // 463: https://leetcode-cn.com/problems/island-perimeter/
    pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
        let mut res = 0;
        let r = grid.len();
        if r == 0 {
            return res;
        }
        let c = grid[0].len();
        for i in 0..r {
            for j in 0..c {
                if grid[i][j] == 0 {
                    continue;
                }
                if i < r - 1 && grid[i + 1][j] == 0 || i == r - 1 {
                    res += 1;
                }
                if j < c - 1 && grid[i][j + 1] == 0 || j == c - 1 {
                    res += 1;
                }
                if i > 0 && grid[i - 1][j] == 0 || i == 0 {
                    res += 1;
                }
                if j > 0 && grid[i][j - 1] == 0 || j == 0 {
                    res += 1;
                }
            }
        }
        res
    }
    // 485: https://leetcode-cn.com/problems/max-consecutive-ones/
    pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        if *nums.last().unwrap() == 1 {
            nums.push(0);
        }
        let mut res = 0;
        let mut start = -1 as i32;
        for (i, n) in nums.iter().enumerate() {
            if *n == 0 {
                res = cmp::max(res, i as i32 - start - 1);
                start = i as i32;
            }
        }
        res
    }
    // 495: https://leetcode-cn.com/problems/teemo-attacking/
    pub fn find_poisoned_duration(time_series: Vec<i32>, duration: i32) -> i32 {
        let mut res = duration;
        for i in 0..time_series.len() - 1 {
            res += cmp::min(duration, time_series[i + 1] - time_series[i]);
        }
        res
    }
    //496: https://leetcode.cn/problems/next-greater-element-i/
    pub fn next_greater_element(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        let mut nums2 = nums2;
        let mut map = HashMap::new();
        for i in 0..nums2.len() {
            map.insert(nums2[i], i);
        }
        let mut stack = Vec::new();
        for i in (0..nums2.len()).rev() {
            let curr = nums2[i];
            while stack.len() > 0 && *stack.last().unwrap() < curr {
                stack.pop();
            }
            nums2[i] = match stack.last() {
                Some(&num) => num,
                None => -1,
            };
            stack.push(curr);
        }
        let mut res = Vec::new();
        for num in nums1 {
            res.push(nums2[*map.get(&num).unwrap()])
        }
        res
    }
    // 500: https://leetcode.cn/problems/keyboard-row/
    pub fn find_words(words: Vec<String>) -> Vec<String> {
        let map: HashMap<char, i32> = vec![
            ('q', 0),
            ('w', 0),
            ('e', 0),
            ('r', 0),
            ('t', 0),
            ('y', 0),
            ('u', 0),
            ('i', 0),
            ('o', 0),
            ('p', 0),
            ('a', 1),
            ('s', 1),
            ('d', 1),
            ('f', 1),
            ('g', 1),
            ('h', 1),
            ('j', 1),
            ('k', 1),
            ('l', 1),
            ('z', 2),
            ('x', 2),
            ('c', 2),
            ('v', 2),
            ('b', 2),
            ('n', 2),
            ('m', 2),
        ]
        .into_iter()
        .collect();

        let mut res = Vec::new();
        for word in words {
            let w = word.to_lowercase();
            let mut line = *map.get(&w.chars().next().unwrap()).unwrap();
            for c in w.chars() {
                if *map.get(&c).unwrap() != line {
                    line = -1;
                    break;
                }
            }
            if line != -1 {
                res.push(word);
            }
        }
        res
    }
    // unsolverd
    // 506: https://leetcode-cn.com/problems/relative-ranks/
    pub fn find_relative_ranks(nums: Vec<i32>) -> Vec<String> {
        let mut sorted = nums.clone();
        sorted.sort();
        let mut map = HashMap::new();
        for i in 0..sorted.len() {
            map.insert(sorted[i], i);
        }
        let mut res = Vec::new();
        for num in nums {
            let s = match *map.get(&num).unwrap() {
                1 => String::from("Gold Medal"),
                2 => String::from("Silver Medal"),
                3 => String::from("Bronze Medal"),
                num => num.to_string(),
            };
            res.push(s);
        }
        res
    }
    // 561: https://leetcode-cn.com/problems/array-partition-i/
    pub fn array_partition_i(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        nums.sort();
        let mut res = 0;
        for &num in nums.iter().step_by(2) {
            res += num;
        }
        res
    }
    // 566: https://leetcode-cn.com/problems/reshape-the-matrix/
    pub fn matrix_reshape(nums: Vec<Vec<i32>>, r: i32, c: i32) -> Vec<Vec<i32>> {
        let mut res = vec![vec![0; c as usize]; r as usize];
        let pr = nums.len();
        let r = r as usize;
        let cnt = pr * nums[0].len();
        for i in 0..cnt {
            res[i/r][i%r] = nums[i/pr][i%pr];
        }
        res
    }
    // 575: https://leetcode-cn.com/problems/distribute-candies/
    pub fn distribute_candies(candies: Vec<i32>) -> i32 {
        let len = candies.len();
        let set: HashSet<_> = candies.into_iter().collect();
        return cmp::min(set.len(), len / 2) as i32;
    }
    // 594: https://leetcode-cn.com/problems/longest-harmonious-subsequence/
    pub fn find_lhs(nums: Vec<i32>) -> i32 {
        let mut map = HashMap::new();
        for num in nums {
            *map.entry(num).or_insert(0) += 1;
        }
        let mut max = 0;
        for (key, value) in map {
            max = cmp::max(value + *map.entry(key + 1).or_default(), max);
        }
        max
    }
}
