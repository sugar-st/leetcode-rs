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
        } else if i > 0 && nums[i] > -nums[i - 1] {
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
    let mut state = 0;
    let mut count = 0;
    for i in 0..nums.len() {
        if state < sum / 3 {
            state += nums[i];
        } else if state == sum / 3 {
            state = 0;
            count += 1;
            if count == 2 {
                break;
            }
        } else {
            return false;
        }
    }
    true
}
// 1018: https://leetcode.cn/problems/binary-prefix-divisible-by-5/
pub fn prefixes_div_by5(nums: Vec<i32>) -> Vec<bool> {
    // simulation
    nums.iter()
        .fold(
            (Vec::with_capacity(nums.len()), 0),
            |(mut res, mut num), &x| {
                num = num * 2 + x;
                res.push(num % 5 == 0);
                (res, num)
            },
        )
        .0
}
// 1030: https://leetcode.cn/problems/matrix-cells-in-distance-order/
pub fn all_cells_dist_order(rows: i32, cols: i32, r_center: i32, c_center: i32) -> Vec<Vec<i32>> {
    panic!("");
}
// 1037: https://leetcode.cn/problems/valid-boomerang/
pub fn is_boomerang(p: Vec<Vec<i32>>) -> bool {
    // math: vector(ab) x vector(bc) != 0
    (p[0][0] - p[1][0]) * (p[0][1] - p[2][1]) != (p[0][0] - p[2][0]) * (p[0][1] - p[1][1])
}
// 1046: https://leetcode.cn/problems/last-stone-weight/
pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
    panic!("")
}
// 1051: https://leetcode.cn/problems/height-checker/
pub fn height_checker(heights: Vec<i32>) -> i32 {
    // sort and compare
    let mut sorted = heights.clone();
    sorted.sort();
    sorted.into_iter().zip(heights.into_iter()).fold(
        0,
        |res, (new, old)| {
            if new != old {
                res + 1
            } else {
                res
            }
        },
    )
}
// 1089: https://leetcode.cn/problems/duplicate-zeros/
pub fn duplicate_zeros(nums: &mut Vec<i32>) {
    // simulation started from behind
    let mut cnt = 0;
    for ele in nums {
        if *ele == 0 {
            cnt += 2;
        }
    }
    for i in (0..nums.len()).rev() {
        if nums[i] == 0 {
            cnt -= 2;
        }
    }
    panic!("");
}
