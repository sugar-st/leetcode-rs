#![allow(dead_code)]
#![allow(unused_variables)]

// LCS 01: https://leetcode.cn/problems/Ju9Xwi/
pub fn least_minutes(n: i32) -> i32 {
    let mut res = n;
    let mut speed = 1;
    let mut upgrade_time = 0;
    let mut download_time = 0;
    while speed <= n {
        speed <<= 1;
        upgrade_time += 1;
        download_time = n / speed;
        if n % speed != 0 {
            download_time += 1;
        }
        res = res.min(upgrade_time + download_time);
    }
    res
}