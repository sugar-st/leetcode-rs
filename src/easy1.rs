#![allow(dead_code)]
#![allow(unused_variables)]

use std::cell::RefCell;
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

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

// 1: https://leetcode-cn.com/problems/two-sum/
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    use std::collections::HashMap;
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
// 9: https://leetcode.cn/problems/palindrome-number/
pub fn is_palindrome(x: i32) -> bool {
    if x < 0 {
        return false;
    }
    let mut cpy = x as i64;
    let mut dest = 0;
    while cpy > 0 {
        let digit = cpy % 10;
        cpy /= 10;
        dest = dest * 10 + digit;
    }
    x == dest as i32
}
// 13: https://leetcode.cn/problems/roman-to-integer/
pub fn roman_to_int(s: String) -> i32 {
    let s = s.as_bytes();
    let mut i = 0;
    let mut res = 0;
    let value = |a: u8| match a {
        b'I' => 1,
        b'V' => 5,
        b'X' => 10,
        b'L' => 50,
        b'C' => 100,
        b'D' => 500,
        b'M' => 1000,
        _ => 0,
    };
    let template = |a: u8, b: u8, c: u8, i: &mut usize| {
        let res;
        if *i < s.len() - 1 && (s[*i + 1] == b || s[*i + 1] == c) {
            if s[*i + 1] == b {
                res = value(b) - value(a);
            } else {
                res = value(c) - value(a);
            }
            *i += 1;
        } else {
            res = value(a);
        }
        res
    };
    while i < s.len() {
        match s[i] {
            b'I' => res += template(b'I', b'V', b'X', &mut i),
            b'X' => res += template(b'X', b'L', b'C', &mut i),
            b'C' => res += template(b'C', b'D', b'M', &mut i),
            a => res += value(a),
        }
        i += 1;
    }
    res
}
// 14: https://leetcode.cn/problems/longest-common-prefix/
// find the max str and min str, and get common prefix between them
pub fn longest_common_prefix(strs: Vec<String>) -> String {
    strs.iter()
        .max()
        .unwrap()
        .chars()
        .zip(strs.iter().min().unwrap().chars())
        .take_while(|x| x.0 == x.1)
        .map(|x| x.0)
        .collect()
}
// 20: https://leetcode.cn/problems/valid-parentheses/
pub fn is_valid(s: String) -> bool {
    if s.len() % 2 != 0 {
        return false;
    }
    let mut stack = vec!['a'];

    for c in s.chars() {
        match c {
            '(' | '[' | '{' => stack.push(c),
            ')' => {
                if stack.pop().unwrap() != '(' {
                    return false;
                }
            }
            ']' => {
                if stack.pop().unwrap() != '[' {
                    return false;
                }
            }
            '}' => {
                if stack.pop().unwrap() != '{' {
                    return false;
                }
            }
            _ => return false,
        }
    }
    stack.len() == 1
}
// 21: https://leetcode.cn/problems/merge-two-sorted-lists/
pub fn merge_two_lists(
    mut list1: Option<Box<ListNode>>,
    mut list2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut dummy = Some(Box::new(ListNode::new(0)));
    let mut curr = &mut dummy;
    loop {
        match (&list1, &list2) {
            (Some(v1), Some(v2)) => {
                if v1.val < v2.val {
                    curr.as_mut().unwrap().next = Some(v1.to_owned());
                    list1 = list1.unwrap().next;
                } else {
                    curr.as_mut().unwrap().next = Some(v2.to_owned());
                    list2 = list2.unwrap().next;
                }
                curr = &mut curr.as_mut().unwrap().next;
            }
            (None, Some(v2)) => {
                curr.as_mut().unwrap().next = Some(v2.to_owned());
                break;
            }
            (Some(v1), None) => {
                curr.as_mut().unwrap().next = Some(v1.to_owned());
                break;
            }
            (None, None) => break,
        }
    }
    dummy.unwrap().next
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
// 29: https://leetcode.cn/problems/implement-strstr/
pub fn str_str(haystack: String, needle: String) -> i32 {
    if haystack.len() < needle.len() {
        return -1;
    }
    let mut haystack = haystack.as_bytes();
    let needle = needle.as_bytes();
    for i in 0..(haystack.len() - needle.len() + 1) {
        if haystack
            .iter()
            .zip(needle.iter())
            .take_while(|x| x.0 == x.1)
            .count()
            == needle.len()
        {
            return i as i32;
        }
        if haystack.len() == 0 {
            break;
        }
        haystack = &haystack[1..];
    }
    -1
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
        res = sum.max(res);
    }
    return res;
}
// 58: https://leetcode.cn/problems/length-of-last-word/
pub fn length_of_last_word(s: String) -> i32 {
    s.chars()
        .rev()
        .skip_while(|x| x.is_whitespace())
        .take_while(|x| !x.is_whitespace())
        .count() as i32
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
// 67: https://leetcode.cn/problems/add-binary/
pub fn add_binary(a: String, b: String) -> String {
    let mut carry = 0;
    let l = a.len().max(b.len());
    let mut a = a.as_bytes().iter().rev();
    let mut b = b.as_bytes().iter().rev();
    let mut res = vec![b'0'; l];
    for i in 1..=l {
        let x = a.next().unwrap_or(&b'0');
        let y = b.next().unwrap_or(&b'0');
        res[l - i] = x + y - b'0' + carry;
        match res[l - i] {
            b'3' => {
                res[l - i] = b'1';
                carry = 1;
            }
            b'2' => {
                res[l - i] = b'0';
                carry = 1;
            }
            _ => {
                carry = 0;
            }
        }
    }
    if carry == 1 {
        res.insert(0, b'1');
    }
    String::from_utf8(res).unwrap()
}
// 69: https://leetcode.cn/problems/sqrtx/
pub fn my_sqrt(x: i32) -> i32 {
    let x = x as i64;
    let (mut left, mut right) = (0, x);
    while left <= right {
        let mid = left + (right - left) / 2;
        let square = mid * mid;
        if square == x || square < x && (mid + 1) * (mid + 1) > x {
            return mid as i32;
        } else if square > x {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    panic!("should't reach here")
}
// 70: https://leetcode.cn/problems/climbing-stairs/
pub fn climb_stairs(n: i32) -> i32 {
    let mut n = n;
    let mut a = vec![1, 1];
    while n != 1 {
        a[1] = a[0] + a[1];
        a[0] = a[1] - a[0];
        n -= 1;
    }
    a[1]
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
        res = res.max(max - prices[i]);
        max = max.max(&prices[i]);
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
    use std::collections::HashSet;
    let set: HashSet<i32> = nums.clone().into_iter().collect();
    set.len() != nums.len()
}
// 219: https://leetcode-cn.com/problems/contains-duplicate-ii/
// slide-window suits here. although the algorithm seems more delicate but the implementation
// is more complicated.
pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
    use std::collections::HashMap;
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

//303: https://leetcode-cn.com/problems/range-sum-query-immutable/
struct NumArray {
    sums: Vec<i32>,
}

impl NumArray {
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

    fn sum_range(&self, left: i32, right: i32) -> i32 {
        let sum = if left == 0 {
            0
        } else {
            self.sums[left as usize - 1]
        };
        self.sums[right as usize] - sum
    }
}
// 349: https://leetcode-cn.com/problems/intersection-of-two-arrays/
pub fn intersection(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashSet;
    nums1
        .into_iter()
        .collect::<HashSet<i32>>()
        .intersection(&nums2.into_iter().collect::<HashSet<i32>>())
        .map(|&num| num)
        .collect()
}
// 350: https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/
pub fn intersect(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashMap;
    fn frequence(nums: &Vec<i32>) -> HashMap<i32, usize> {
        nums.iter().fold(HashMap::new(), |mut m, num| {
            *m.entry(*num).or_insert(0) += 1;
            m
        })
    }
    let m1 = frequence(&nums1);
    let m2 = frequence(&nums2);
    m1.iter().fold(Vec::new(), |mut res, (&num, &freq)| {
        res.append(&mut vec![num; freq.min(*m2.get(&num).unwrap_or(&0))]);
        res
    })
}
// 409: https://leetcode.cn/problems/longest-palindrome/
pub fn longest_palindrome(s: String) -> i32 {
    let s = s.as_bytes().iter();
    use std::collections::HashMap;
    let mut has_odd = false;
    s.fold(HashMap::new(), |mut m, x| {
        *m.entry(x).or_insert(0) += 1;
        m
    })
    .iter()
    .fold(0, |res, (_, v)| {
        if v % 2 == 0 {
            res + v
        } else if has_odd {
            res + v - 1
        } else {
            has_odd = true;
            res + v
        }
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
pub fn find_content_children(mut g: Vec<i32>, mut s: Vec<i32>) -> i32 {
    s.sort();
    g.sort();
    let mut res = 0;
    let (mut i, mut j) = (0, 0);
    while i < g.len() && j < s.len() {
        if g[i] <= s[j] {
            res += 1;
            i += 1; 
        }
        j += 1;
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
            res = res.max(i as i32 - start - 1);
            start = i as i32;
        }
    }
    res
}
// 495: https://leetcode-cn.com/problems/teemo-attacking/
pub fn find_poisoned_duration(time_series: Vec<i32>, duration: i32) -> i32 {
    let mut res = duration;
    for i in 0..time_series.len() - 1 {
        res += duration.min(time_series[i + 1] - time_series[i]);
    }
    res
}
//496: https://leetcode.cn/problems/next-greater-element-i/
pub fn next_greater_element(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashMap;
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
    use std::collections::HashMap;
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
// 506: https://leetcode-cn.com/problems/relative-ranks/
pub fn find_relative_ranks(nums: Vec<i32>) -> Vec<String> {
    use std::collections::HashMap;
    let mut sorted = nums.clone();
    sorted.sort();
    sorted.reverse();
    let mut map = HashMap::new();
    for i in 0..sorted.len() {
        map.insert(sorted[i], i + 1);
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
pub fn array_pair_sum(mut nums: Vec<i32>) -> i32 {
    nums.sort();
    nums.iter().step_by(2).sum()
}
// 566: https://leetcode-cn.com/problems/reshape-the-matrix/
pub fn matrix_reshape(nums: Vec<Vec<i32>>, r: i32, c: i32) -> Vec<Vec<i32>> {
    if nums[0].len() * nums.len() != (r * c) as usize {
        return nums;
    }
    nums.concat()
        .chunks(c as usize)
        .map(|x| x.to_vec())
        .collect()
}
// 575: https://leetcode-cn.com/problems/distribute-candies/
pub fn distribute_candies(candies: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let len = candies.len();
    let set: HashSet<_> = candies.into_iter().collect();
    set.len().min(len / 2) as i32
}
// 594: https://leetcode-cn.com/problems/longest-harmonious-subsequence/
pub fn find_lhs(nums: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    for num in nums {
        *map.entry(num).or_insert(0) += 1;
    }
    let mut max = 0;
    for (&key, &value) in map.iter() {
        if let Some(&v) = map.get(&(key + 1)) {
            max = max.max(value + v);
        }
    }
    max
}
// 598: https://leetcode-cn.com/problems/range-addition-ii/
pub fn max_count(m: i32, n: i32, ops: Vec<Vec<i32>>) -> i32 {
    let mut x = m;
    let mut y = n;
    for op in ops {
        x = x.min(op[0]);
        y = y.min(op[1]);
    }
    x * y
}
// 599: https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/
pub fn find_restaurant(list1: Vec<String>, list2: Vec<String>) -> Vec<String> {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    let mut min = usize::MAX;
    let mut res = Vec::new();
    for i in 0..list1.len() {
        map.insert(&list1[i], i);
    }
    for i in 0..list2.len() {
        if let Some(common) = map.get(&list2[i]) {
            if common + i <= min {
                if common + i < min {
                    res.clear();
                    min = common + i;
                }
                res.push(list2[i].clone());
            }
        }
    }
    res
}
// 605: https://leetcode-cn.com/problems/can-place-flowers/
pub fn can_place_flowers(flowerbed: Vec<i32>, n: i32) -> bool {
    let mut flowerbed = flowerbed;
    let mut max = 0;
    for i in 0..flowerbed.len() {
        if flowerbed[i] == 0 {
            if (i == 0 || i > 0 && flowerbed[i - 1] == 0)
                && (i == flowerbed.len() - 1 || flowerbed[i + 1] == 0)
            {
                flowerbed[i] = 1;
                max += 1;
            }
        }
    }
    n <= max
}
// 628: https://leetcode-cn.com/problems/maximum-product-of-three-numbers/
pub fn maximum_product(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let len = nums.len();
    (nums[0] * nums[1] * nums[len - 1]).max(nums[len - 1] * nums[len - 2] * nums[len - 3])
}
// 643: https://leetcode-cn.com/problems/maximum-average-subarray-i/
pub fn find_max_average(nums: Vec<i32>, k: i32) -> f64 {
    let k = k as usize;
    let mut current = 0;
    for i in 0..k {
        current += nums[i];
    }
    let mut max = current;
    for i in k..nums.len() {
        current += nums[i] - nums[i - k];
        max = max.max(current);
    }
    max as f64 / k as f64
}
// 645: https://leetcode-cn.com/problems/set-mismatch/
pub fn find_error_nums(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let mut res = vec![0, 0];
    for i in 0..nums.len() {
        let num = nums[i].abs();
        let hash = &mut nums[num as usize - 1];
        if *hash < 0 {
            res[0] = num;
        } else {
            *hash = -*hash;
        }
    }
    for i in 0..nums.len() {
        if nums[i] > 0 {
            res[1] = i as i32 + 1;
            break;
        }
    }
    res
}
// 661: https://leetcode-cn.com/problems/image-smoother/
pub fn image_smoother(m: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let r = m.len();
    let c = m[0].len();
    let mut res = vec![vec![0; c]; r];
    let vectors = vec![
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    for i in 0..r {
        for j in 0..c {
            let mut cnt = 0;
            for vector in vectors.iter() {
                let ni = i as i32 + vector.0;
                let nj = j as i32 + vector.1;
                if ni >= 0 && ni < r as i32 && nj >= 0 && nj < c as i32 {
                    cnt += 1;
                    res[i][j] += m[ni as usize][nj as usize];
                }
            }
            res[i][j] /= cnt;
        }
    }
    res
}
// 674: https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/
pub fn find_length_of_lcis(nums: Vec<i32>) -> i32 {
    let mut res = 1;
    let mut curr = 1;
    for i in 0..nums.len() {
        if i + 1 < nums.len() && nums[i + 1] > nums[i] {
            curr += 1;
        } else {
            res = res.max(curr);
            curr = 1;
        }
    }
    res
}
// 682: https://leetcode-cn.com/problems/baseball-game/
pub fn cal_points(ops: Vec<String>) -> i32 {
    ops.iter()
        .fold(Vec::new(), |mut v, op| {
            let len = v.len();
            match op.as_str() {
                "C" => {
                    v.pop();
                }
                "D" => v.push(v[len - 1] * 2),
                "+" => v.push(v[len - 1] + v[len - 2]),
                num => v.push(num.parse().unwrap()),
            }
            v
        })
        .iter()
        .sum()
}
// 697: https://leetcode-cn.com/problems/degree-of-an-array/
pub fn find_shortest_sub_array(nums: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    let mut list = Vec::new();
    let mut max = 0;
    for (i, num) in nums.iter().enumerate() {
        let entry = map.entry(num).or_insert((0, i, i));
        entry.0 += 1;
        entry.2 = i;
        if max == entry.0 {
            list.push(num);
        } else if entry.0 > max {
            max = entry.0;
            list.clear();
            list.push(num);
        }
    }
    let mut min = usize::MAX;
    for num in list {
        let entry = map.get(num).unwrap();
        min = min.min(entry.2 - entry.1 + 1);
    }
    min as i32
}
// 704: https://leetcode-cn.com/problems/binary-search/
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    *nums.iter().find(|&&x| x == target).unwrap_or(&-1)
}

// 705: https://leetcode.cn/problems/design-hashset/
struct MyHashSet {}

impl MyHashSet {
    fn new() -> Self {
        panic!("");
    }

    fn add(&self, _key: i32) {}

    fn remove(&self, _key: i32) {}

    fn contains(&self, _key: i32) -> bool {
        panic!("");
    }
}
// 706: https://leetcode.cn/problems/design-hashmap/
struct MyHashMap {}

impl MyHashMap {
    fn new() -> Self {
        panic!("");
    }

    fn put(&self, _key: i32, _value: i32) {}

    fn get(&self, _key: i32) -> i32 {
        panic!("");
    }

    fn remove(&self, _key: i32) {}
}
// 717: https://leetcode-cn.com/problems/1-bit-and-2-bit-characters/
pub fn is_one_bit_character(bits: Vec<i32>) -> bool {
    let mut i = 0;
    while i < bits.len() - 1 {
        i += bits[i] as usize + 1;
    }
    i == bits.len() - 1
}
// 720: https://leetcode-cn.com/problems/longest-word-in-dictionary/
pub fn longest_word(words: Vec<String>) -> String {
    use std::collections::HashSet;
    let mut words = words;
    words.sort();
    words
        .iter()
        .fold(("", HashSet::new()), |(mut res, mut set), str| {
            if str.len() == 1 || set.contains(&str[..str.len() - 1]) {
                set.insert(str.as_str());
                if res.len() < str.len() {
                    res = str;
                }
            }
            (res, set)
        })
        .0
        .to_string()
}
// 724: https://leetcode-cn.com/problems/find-pivot-index/
pub fn pivot_index(nums: Vec<i32>) -> i32 {
    let mut state = 0;
    let sum = nums.iter().sum::<i32>();
    for i in 0..nums.len() {
        if sum == state * 2 + nums[i] {
            return i as i32;
        }
        state += nums[i];
    }
    -1
}
// 733: https://leetcode-cn.com/problems/flood-fill/
pub fn flood_fill(image: Vec<Vec<i32>>, sr: i32, sc: i32, new_color: i32) -> Vec<Vec<i32>> {
    let mut image = image;
    let origin = &mut image[sr as usize][sc as usize];
    let old_color = *origin;
    if old_color != new_color {
        *origin = new_color;
        for (i, j) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (dr, dc) = ((sr + i) as usize, (sc + j) as usize);
            if dr < image.len() && dc < image[0].len() && old_color == image[dr][dc] {
                image = flood_fill(image, dr as i32, dc as i32, new_color)
            }
        }
    }
    image
}
// 744: https://leetcode.cn/problems/find-smallest-letter-greater-than-target/
pub fn next_greatest_letter(letters: Vec<char>, target: char) -> char {
    let (mut s, mut e) = (0, letters.len());
    while s < e {
        let m = s + (e - s) / 2;
        if letters[m] <= target {
            s = m + 1;
        } else {
            e = m;
        }
    }
    if s < letters.len() && letters[s] > target {
        return letters[s];
    } else {
        letters[0]
    }
}
// 747: https://leetcode.cn/problems/largest-number-at-least-twice-of-others/
pub fn dominant_index(nums: Vec<i32>) -> i32 {
    let max = nums
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.cmp(y.1))
        .unwrap()
        .0;
    for i in 0..nums.len() {
        if i == max {
            continue;
        }
        if nums[i] * 2 > nums[max] {
            return -1;
        }
    }
    max as i32
}
// 748: https://leetcode.cn/problems/shortest-completing-word/
pub fn shortest_completing_word(license_plate: String, words: Vec<String>) -> String {
    fn collect(word: &String) -> Vec<i32> {
        word.bytes().fold(vec![0; 26], |mut map, c| {
            if c.is_ascii_alphabetic() {
                map[(c.to_ascii_lowercase() - 'a' as u8) as usize] += 1;
            }
            map
        })
    }
    fn cmp(x: &Vec<i32>, y: &Vec<i32>) -> bool {
        for i in 0..26 {
            if y[i] < x[i] {
                return false;
            }
        }
        true
    }
    let mut ans: Option<String> = None;
    let license_plate = collect(&license_plate);
    for word in words {
        if cmp(&license_plate, &collect(&word)) {
            if ans.is_some() {
                if ans.as_ref().unwrap().bytes().len() > word.bytes().len() {
                    ans = Some(word);
                }
            } else {
                ans = Some(word);
            }
        }
    }
    ans.unwrap()
}

// 766: https://leetcode.cn/problems/toeplitz-matrix/
pub fn is_toeplitz_matrix(matrix: Vec<Vec<i32>>) -> bool {
    for i in 0..(matrix.len() - 1) {
        for j in 0..(matrix[0].len() - 1) {
            if matrix[i][j] != matrix[i + 1][j + 1] {
                return false;
            }
        }
    }
    true
}
// 806: https://leetcode.cn/problems/number-of-lines-to-write-string/
pub fn number_of_lines(widths: Vec<i32>, s: String) -> Vec<i32> {
    let mut res = vec![1, 0];
    for c in s.bytes() {
        let idx = c as usize - 'a' as usize;
        if widths[idx] + res[1] > 100 {
            res[1] = widths[idx];
            res[0] += 1;
        } else {
            res[1] += widths[idx];
        }
    }
    res
}
// 804: https://leetcode.cn/problems/unique-morse-code-words/
pub fn unique_morse_representations(words: Vec<String>) -> i32 {
    use std::collections::HashSet;
    let code = vec![
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
        "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--",
        "--..",
    ];
    let mut map = HashSet::new();
    for word in words {
        let mut str = String::from("");
        for c in word.bytes() {
            str.push_str(code[(c - 'a' as u8) as usize]);
        }
        map.insert(str);
    }
    map.len() as i32
}
// 812: https://leetcode.cn/problems/largest-triangle-area/
pub fn largest_triangle_area(points: Vec<Vec<i32>>) -> f64 {
    panic!("");
}
// 821: https://leetcode.cn/problems/shortest-distance-to-a-character/
pub fn shortest_to_char(s: String, c: char) -> Vec<i32> {
    let mut res = vec![std::i32::MAX; s.len()];
    for (idx, ele) in s.chars().enumerate() {
        if ele != c {
            continue;
        }
        res[idx] = 0;
        let mut i = idx - 1;
        while i < res.len() && res[i] >= (idx - i) as i32 {
            res[i] = (idx - i) as i32;
            i -= 1;
        }
        i = idx + 1;
        while i < res.len() && res[i] >= (i - idx) as i32 {
            res[i] = (i - idx) as i32;
            i += 1;
        }
    }
    res
}
// 832: https://leetcode.cn/problems/flipping-an-image
pub fn flip_and_invert_image(image: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut image = image;
    for line in image.iter_mut() {
        line.reverse();
        for e in line.iter_mut() {
            if *e == 0 {
                *e = 1;
            } else {
                *e = 0;
            }
        }
    }
    image
}
// 852: https://leetcode.cn/problems/peak-index-in-a-mountain-array/
pub fn peak_index_in_mountain_array(arr: Vec<i32>) -> i32 {
    let mut left = 1;
    let mut right = arr.len() - 1;
    while left < right {
        let mid = left + (right - left) / 2;
        if arr[mid] > arr[mid - 1] {
            if arr[mid] > arr[mid + 1] {
                return mid as i32;
            } else {
                left = mid + 1;
            }
        } else {
            right = mid;
        }
    }
    panic!("not a mountain");
}
// 860: https://leetcode.cn/problems/lemonade-change/
pub fn lemonade_change(bills: Vec<i32>) -> bool {
    let mut five = 0;
    let mut ten = 0;
    for bill in bills {
        match bill {
            5 => {
                five += 1;
            }
            10 => {
                five -= 1;
                ten += 1;
            }
            20 => {
                if ten > 0 {
                    ten -= 1;
                    five -= 1;
                } else {
                    five -= 3;
                }
            }
            _ => {
                panic!("error");
            }
        }
        if five < 0 || ten < 0 {
            return false;
        }
    }
    true
}
// 867: https://leetcode.cn/problems/transpose-matrix/
pub fn transpose(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut res = vec![vec![0; matrix.len()]; matrix[0].len()];
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            res[j][i] = matrix[i][j];
        }
    }
    res
}
// 883: https://leetcode.cn/problems/projection-area-of-3d-shapes/
pub fn projection_area(grid: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    let mut front = vec![0; grid.len()];
    for i in 0..grid.len() {
        let mut max = 0;
        for j in 0..grid[0].len() {
            // top
            if grid[i][j] != 0 {
                res += 1;
            }
            //side
            max = max.max(grid[i][j]);
            //front
            front[j] = front[j].max(grid[i][j]);
        }
        res += max;
    }
    for front in front {
        res += front;
    }
    res
}
// 888: https://leetcode.cn/problems/fair-candy-swap/
pub fn fair_candy_swap(alice_sizes: Vec<i32>, bob_sizes: Vec<i32>) -> Vec<i32> {
    let mut alice_sizes = alice_sizes;
    alice_sizes.sort();
    let mut bob_sizes = bob_sizes;
    bob_sizes.sort();
    let am = alice_sizes.iter().sum::<i32>();
    let bm = bob_sizes.iter().sum::<i32>();
    let sub = (am + bm) / 2 - am;
    let (mut i, mut j) = (0, 0);
    while i < alice_sizes.len() && j < bob_sizes.len() {
        let ac = alice_sizes[i];
        let bc = bob_sizes[j];
        if ac + sub == bc {
            return vec![ac, bc];
        } else if ac + sub < bc {
            i += 1;
        } else {
            j += 1;
        }
    }
    panic!("not found");
}
// 892: https://leetcode.cn/problems/surface-area-of-3d-shapes/
pub fn surface_area(grid: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    let (row, col) = (grid.len(), grid[0].len());
    for i in 0..row {
        for j in 0..col {
            if grid[i][j] != 0 {
                res += 2;
            }
            for (x, y) in vec![(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)] {
                if x < row && y < col && grid[x][y] < grid[i][j] {
                    res += grid[i][j] - grid[x][y];
                } else if x >= row || y >= col {
                    res += grid[i][j]
                }
            }
        }
    }
    res
}
// 896: https://leetcode.cn/problems/monotonic-array/
pub fn is_monotonic(nums: Vec<i32>) -> bool {
    let mut up = false;
    let mut down = false;
    for i in 1..nums.len() {
        if nums[i] > nums[i - 1] {
            up = true;
        } else if nums[i] < nums[i - 1] {
            down = true;
        }
    }
    !(up && down)
}
// 905: https://leetcode.cn/problems/sort-array-by-parity/
pub fn sort_array_by_parity(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let mut last = 0;
    for i in 0..nums.len() {
        if nums[i] % 2 == 0 {
            let tmp = nums[last];
            nums[last] = nums[i];
            nums[i] = tmp;
            last += 1;
        }
    }
    nums
}
// 908: https://leetcode.cn/problems/smallest-range-i/
pub fn smallest_range_i(nums: Vec<i32>, k: i32) -> i32 {
    let (mut min, mut max) = (std::i32::MAX, std::i32::MIN);
    for num in nums {
        min = min.min(num);
        max = max.max(num);
    }
    let distance = max - min - 2 * k;
    if distance < 0 {
        0
    } else {
        distance
    }
}
// 914: https://leetcode.cn/problems/x-of-a-kind-in-a-deck-of-cards/
pub fn has_groups_size_x(deck: Vec<i32>) -> bool {
    use std::collections::{HashMap, HashSet};
    let cnts = deck
        .into_iter()
        .fold(HashMap::new(), |mut m, x| {
            *m.entry(x).or_insert(0) += 1;
            m
        })
        .into_iter()
        .fold(HashSet::new(), |mut s, (_, count)| {
            s.insert(count);
            s
        })
        .into_iter()
        .collect::<Vec<i32>>();
    if cnts.len() == 1 && cnts[0] > 1 {
        return true;
    }
    cnts.iter().fold(cnts[0], |res, &num| {
        if res <= 1 || num == res {
            return res;
        }
        fn gcd(i: i32, j: i32) -> i32 {
            match (i, j) {
                (0, 0) => {
                    panic!("panic");
                }
                (0, a) => a,
                _ => gcd(j % i, i),
            }
        }
        gcd(res, num)
    }) > 1
}
// 922: https://leetcode.cn/problems/sort-array-by-parity-ii/
pub fn sort_array_by_parity_ii(nums: Vec<i32>) -> Vec<i32> {
    // double pointers: swap until either of pointer reach the lenth;
    let mut nums = nums;
    let (mut i, mut j) = (0, 1);
    while i < nums.len() && j < nums.len() {
        if nums[i] % 2 != 0 {
            let tmp = nums[j];
            nums[j] = nums[i];
            nums[i] = tmp;
            j += 2;
        } else {
            i += 2;
        }
    }
    nums
}
// 929: https://leetcode.cn/problems/unique-email-addresses/solution/
pub fn num_unique_emails(emails: Vec<String>) -> i32 {
    // simplify emails and count the length of set
    use std::collections::HashSet;
    emails
        .iter()
        .map(|str| {
            let mut strs = str.split('@');
            let local = strs
                .next()
                .unwrap()
                .split('+')
                .next()
                .unwrap()
                .replace(".", "");
            let remote = strs.next().unwrap();
            (local, remote)
        })
        .collect::<HashSet<_>>()
        .len() as i32
}
// 937: https://leetcode.cn/problems/reorder-data-in-log-files/
pub fn reorder_log_files(logs: Vec<String>) -> Vec<String> {
    // custom sort
    fn is_digital(s: &str) -> bool {
        for &b in s.as_bytes() {
            if b != b' ' && !(b >= b'0' && b <= b'9') {
                return false;
            }
        }
        true
    }
    let mut logs = logs;
    use std::cmp::Ordering;
    logs.sort_by(|a, b| {
        let (a1, a2) = a.split_once(' ').unwrap();
        let (b1, b2) = b.split_once(' ').unwrap();
        return match (is_digital(a2), is_digital(b2)) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => match a2.partial_cmp(b2).unwrap() {
                Ordering::Equal => a1.partial_cmp(b1).unwrap(),
                order => order,
            },
        };
    });
    logs
}
// 941: https://leetcode.cn/problems/valid-mountain-array/
pub fn valid_mountain_array(arr: Vec<i32>) -> bool {
    // if the arr is not a mountain, there must be a point where the curve like '-' or 'V' rather
    // than '/', '\' or '^'
    let len = arr.len();
    if len < 3 || !(arr[0] < arr[1] && arr[len - 2] > arr[len - 1]) {
        return false;
    }
    for i in 1..(len - 1) {
        if arr[i] == arr[i - 1] || arr[i] <= arr[i - 1] && arr[i] <= arr[i + 1] {
            return false;
        }
    }
    return true;
}
// 942: https://leetcode.cn/problems/di-string-match/
pub fn di_string_match(s: String) -> Vec<i32> {
    // double pointers, greedy
    let (mut start, mut end) = (0, s.len() as i32);
    let mut res = Vec::with_capacity(s.len());
    for c in s.as_bytes() {
        match c {
            b'I' => {
                res.push(start);
                start += 1;
            }
            b'D' => {
                res.push(end);
                end -= 1;
            }
            _ => panic!("invalid"),
        }
    }
    res.push(start);
    res
}
// 944: https://leetcode.cn/problems/delete-columns-to-make-sorted/
pub fn min_deletion_size(strs: Vec<String>) -> i32 {
    // simulation
    let mut res = 0;
    for i in 0..strs[0].len() {
        for j in 1..strs.len() {
            if strs[j].as_bytes()[i] < strs[j - 1].as_bytes()[i] {
                res += 1;
                break;
            }
        }
    }
    res
}
// 953: https://leetcode.cn/problems/verifying-an-alien-dictionary/
pub fn is_alien_sorted(words: Vec<String>, order: String) -> bool {
    // build map and check one by one
    let dict = order
        .as_bytes()
        .into_iter()
        .enumerate()
        .fold(vec![0; 26], |mut dict, (idx, &c)| {
            dict[(c - b'a') as usize] = idx;
            dict
        });
    for i in 1..words.len() {
        let mut flag = 0;
        {
            let cur = words[i].as_bytes();
            let pre = words[i - 1].as_bytes();
            let mut i = 0;
            use std::cmp::Ordering;
            while i < cur.len() && i < pre.len() {
                match dict[(cur[i] - b'a') as usize]
                    .partial_cmp(&dict[(pre[i] - b'a') as usize])
                    .unwrap()
                {
                    Ordering::Less => {
                        flag = -1;
                        break;
                    }
                    Ordering::Greater => {
                        flag = 1;
                        break;
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        if flag == 0 && words[i - 1].len() > words[i].len() || flag == -1 {
            return false;
        }
    }
    true
}
// 961: https://leetcode.cn/problems/n-repeated-element-in-size-2n-array/
pub fn repeated_n_times(nums: Vec<i32>) -> i32 {
    // hashset
    let mut s = std::collections::HashSet::new();
    for num in nums {
        if s.contains(&num) {
            return num;
        } else {
            s.insert(num);
        }
    }
    panic!("invalid");
}
// 976: https://leetcode.cn/problems/largest-perimeter-triangle/
pub fn largest_perimeter(nums: Vec<i32>) -> i32 {
    // greedy
    let mut nums = nums;
    nums.sort();
    for i in (2..nums.len()).rev() {
        if nums[i - 1] + nums[i - 2] > nums[i] {
            return nums[i - 1] + nums[i - 2] + nums[i];
        }
    }
    0
}
// 977: https://leetcode.cn/problems/squares-of-a-sorted-array/
pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
    // double pointers
    let mut last = nums.len() - 1;
    let mut res = vec![0; last + 1];
    let (mut l, mut r) = (0, last);
    while last < nums.len() {
        if nums[l].abs() > nums[r].abs() {
            res[last] = nums[l] * nums[l];
            l += 1;
        } else {
            res[last] = nums[r] * nums[r];
            r -= 1;
        }
        last -= 1;
    }
    res
}
// 989: https://leetcode.cn/problems/add-to-array-form-of-integer/
pub fn add_to_array_form(num: Vec<i32>, k: i32) -> Vec<i32> {
    let mut num = num;
    *num.last_mut().unwrap() += k;
    for i in (0..num.len() - 1).rev() {
        if num[i + 1] <= 9 {
            break;
        }
        num[i] += num[i + 1] / 10;
        num[i + 1] %= 10;
    }
    if num[0] < 10 {
        num
    } else {
        let mut res = Vec::with_capacity(num.len().max(5));
        res.push(num[0] / 10);
        num[0] %= 10;
        let mut cur = 0;
        while res[cur] >= 10 {
            res.push(res[cur] / 10);
            res[cur] %= 10;
            cur += 1;
        }
        res.reverse();
        res.append(&mut num);
        res
    }
}
// 997: https://leetcode.cn/problems/find-the-town-judge/
pub fn find_judge(n: i32, trust: Vec<Vec<i32>>) -> i32 {
    if trust.len() == 0 && n == 1 {
        return 1;
    }
    let m = trust
        .into_iter()
        .fold(vec![(0, 0); n as usize + 1], |mut m, trust| {
            m[trust[1] as usize].0 += 1;
            m[trust[0] as usize].1 += 1;
            m
        });
    for i in 1..m.len() {
        if m[i] == (n - 1, 0) {
            return i as i32;
        }
    }
    -1
}
// 999: https://leetcode.cn/problems/available-captures-for-rook/
pub fn num_rook_captures(board: Vec<Vec<char>>) -> i32 {
    let mut res = 0;
    let (row, col) = (board.len(), board[0].len());
    let mut r = (0, 0);
    for i in 0..row {
        for j in 0..col {
            if board[i][j] == 'R' {
                r = (i, j);
                break;
            }
        }
    }
    for i in (0..r.0).rev() {
        if board[i][r.1] == 'B' {
            break;
        }
        if board[i][r.1] == 'p' {
            res += 1;
            break;
        }
    }
    for i in (r.0 + 1)..row {
        if board[i][r.1] == 'B' {
            break;
        }
        if board[i][r.1] == 'p' {
            res += 1;
            break;
        }
    }
    for j in (0..r.1).rev() {
        if board[r.0][j] == 'B' {
            break;
        }
        if board[r.0][j] == 'p' {
            res += 1;
            break;
        }
    }
    for j in (r.1 + 1)..col {
        if board[r.0][j] == 'B' {
            break;
        }
        if board[r.0][j] == 'p' {
            res += 1;
            break;
        }
    }
    res
}
