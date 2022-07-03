#![allow(dead_code)]
#![allow(unused_variables)]

// 1002: https://leetcode.cn/problems/find-common-characters/
pub fn common_chars(words: Vec<String>) -> Vec<String> {
    fn frequency(word: &String) -> [i8; 26] {
        word.as_bytes().iter().fold([0; 26], |mut freq, &b| {
            freq[b as usize - 'a' as usize] += 1;
            freq
        })
    }
    fn intersection(mut x: [i8; 26], y: [i8; 26]) -> [i8; 26] {
        for i in 0..26 {
            x[i] = x[i].min(y[i]);
        }
        x
    }
    fn form(x: [i8; 26]) -> Vec<String> {
        x.into_iter()
            .enumerate()
            .fold(Vec::new(), |mut res, (idx, count)| {
                for i in 0..count {
                    res.push(String::from_utf8(vec![b'a' + idx as u8]).unwrap());
                }
                res
            })
    }
    let mut freq = frequency(&words[0]);
    for i in 1..words.len() {
        freq = intersection(freq, frequency(&words[i]));
    }
    form(freq)
}
// 1005: https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/
pub fn largest_sum_after_k_negations(nums: Vec<i32>, k: i32) -> i32 {
    // 1. flip negative nums as many as possible
    // 2. flip the minimun abs value until the count is used up
    // 3. sum up the result
    let mut nums = nums;
    let mut k = k as usize;
    nums.sort();
    let mut i = 0;
    while k != 0 && i < nums.len() && nums[i] < 0 {
        nums[i] = -nums[i];
        k -= 1;
        i += 1;
    }
    if k != 0 {
        if i >= nums.len() {
            i = nums.len() - 1;
        } else if i > 0 && nums[i] > nums[i - 1] {
            i = i - 1;
        }
        if k % 2 != 0 {
            nums[i] = -nums[i];
        }
    }
    nums.iter().sum()
}
// 1013: https://leetcode.cn/problems/partition-array-into-three-parts-with-equal-sum/
pub fn can_three_parts_equal_sum(nums: Vec<i32>) -> bool {
    // simulation
    let sum: i32 = nums.iter().sum();
    if sum % 3 != 0 {
        return false;
    }
    let mut state = nums[0]; // at least one element to avoid bug when sum == 0
    let mut count = 0;
    for i in 1..nums.len() {
        if state == sum / 3 {
            state = nums[i];
            count += 1;
            if count == 2 {
                return true;
            }
        } else {
            state += nums[i];
        }
    }
    false
}
// 1018: https://leetcode.cn/problems/binary-prefix-divisible-by-5/
pub fn prefixes_div_by5(nums: Vec<i32>) -> Vec<bool> {
    // simulation, use mod to avoid overflow
    nums.iter()
        .fold(
            (Vec::with_capacity(nums.len()), 0),
            |(mut res, mut num), &x| {
                num = (num * 2 + x) % 5;
                res.push(num == 0);
                (res, num)
            },
        )
        .0
}
