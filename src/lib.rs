use std::cell::RefCell;
use std::rc::Rc;
use std::{
    cmp,
    collections::{HashMap, HashSet},
};

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
                cnt = cnt + 1;
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
    pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {

    }
}
