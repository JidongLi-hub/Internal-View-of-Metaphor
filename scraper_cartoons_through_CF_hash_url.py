"""
Cartoon Movement 图片爬虫 - 接管已有Chrome浏览器版本
"""

import os
import re
import csv
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from DrissionPage import ChromiumPage, ChromiumOptions

"""
首先在windows上启动Chrome浏览器，命令行运行以下脚本，使用远程调试端口9222，这样才可以被接管：
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

其次，在浏览器中安装NopeCHA插件，启用自动点击通过Cloudflare验证功能，以减少人工干预。
"""




class CartoonScraperTakeover:
    def __init__(self, save_dir: str = "downloaded_images", debug_port: int = 9222):
        """
        初始化爬虫 - 接管已有的Chrome浏览器
        
        Args:
            save_dir: 图片保存目录
            debug_port: Chrome调试端口
        """
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://www.cartoonmovement.com"
        self.search_url = "https://www.cartoonmovement.com/search?order=desc&query=&sort=created"
        
        # 配置为接管模式
        self.options = ChromiumOptions()
        self.options.set_local_port(debug_port)  # 连接到已有的调试端口
        
        self.page = None
        self.downloaded_content_hashes = set()  # 图片内容哈希，防止重复内容
        
        # CSV文件路径
        self.csv_path = self.save_dir / "image_index.csv"
        
        # 加载已有记录
        self.image_records = {}  # {title_hash: {'image': filename, 'title': title, 'url': url}}
        self.downloaded_urls = set()  # 已下载的URL集合
        
        self._load_existing_records()
    
    def _get_title_hash(self, title: str, length: int = 12) -> str:
        """
        生成标题的短哈希值
        
        Args:
            title: 图片标题
            length: 哈希值长度（默认12个字符）
        
        Returns:
            短哈希字符串
        """
        # 使用 SHA256 然后截取前 length 个字符
        full_hash = hashlib.sha256(title.encode('utf-8')).hexdigest()
        return full_hash[:length]
        
    def _load_existing_records(self):
        """加载已有的CSV记录"""
        if self.csv_path.exists():
            try:
                with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        title = row['title']
                        image = row['image']
                        url = row.get('url', '')  # 兼容旧版本CSV
                        
                        # 从文件名提取哈希（去掉扩展名）
                        title_hash = os.path.splitext(image)[0]
                        
                        self.image_records[title_hash] = {
                            'image': image,
                            'title': title,
                            'url': url
                        }
                        if url:
                            self.downloaded_urls.add(url)
                    
                print(f"已加载 {len(self.image_records)} 条历史记录")
                
            except Exception as e:
                print(f"加载CSV记录时出错: {e}")
                self.image_records = {}
                self.downloaded_urls = set()
    
    
    def _save_record(self, image_name: str, title: str, url: str):
        """保存一条新记录到CSV"""
        title_hash = os.path.splitext(image_name)[0]
        
        self.image_records[title_hash] = {
            'image': image_name,
            'title': title,
            'url': url
        }
        self.downloaded_urls.add(url)
        
        # 追加写入CSV
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with open(self.csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'title', 'url'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'image': image_name, 'title': title, 'url': url})
    
    def _is_already_downloaded(self, url: str) -> bool:
        """
        检查图片是否已下载
        
        Args:
            url: 图片下载地址
        
        Returns:
            是否已下载
        """
        return url in self.downloaded_urls
    
    def is_cloudflare_challenge(self) -> bool:
        """检测当前页面是否为 Cloudflare 验证页面"""
        try:
            page_title = self.page.title if self.page.title else ""
            
            # 方法1：正向检测 - 如果是正常页面，直接返回 False
            # 正常页面标题包含 "Cartoon Movement"
            if "Cartoon Movement" in page_title:
                return False
            
            # 方法2：精确匹配 Cloudflare 验证页面的标题特征
            cf_title_indicators = [
                # 中文特征（你观察到的）
                "请稍等",
                "请稍候",
                # 英文特征（备用）
                "Just a moment",
                "just a moment",
                "Attention Required",
                "attention required",
            ]
            
            for indicator in cf_title_indicators:
                if indicator in page_title:
                    return True
            
            # 方法3：检查是否存在正常页面的关键元素
            # 如果能找到漫画条目，说明是正常页面
            try:
                articles = self.page.eles('css:article.cartoon--teaser', timeout=2)
                if articles and len(articles) > 0:
                    return False
            except:
                pass
            
            # 方法4：检查 Cloudflare 特有的 DOM 元素（精确匹配）
            try:
                cf_elements = self.page.ele(
                    'css:#challenge-running, css:#challenge-form, css:#challenge-stage, '
                    'css:#turnstile-wrapper, css:iframe[src*="challenges.cloudflare.com"]', 
                    timeout=1
                )
                if cf_elements:
                    return True
            except:
                pass
            
            # 默认返回 False，避免误判
            return False
            
        except Exception as e:
            print(f"检测 Cloudflare 时出错: {e}")
            return False
    
    def wait_for_cloudflare(self, context: str = ""):
        """等待用户完成 Cloudflare 验证"""
        if not self.is_cloudflare_challenge():
            return
        
        print(f"\n{'!'*50}")
        print(f"⚠ 检测到 Cloudflare 验证页面 {f'({context})' if context else ''}")
        print("请在浏览器中手动完成验证...")
        print('!'*50)
        
        # 持续检测，直到验证通过
        check_count = 0
        while self.is_cloudflare_challenge():
            check_count += 1
            if check_count % 10 == 0:  # 每10秒提醒一次
                print(f"  ... 仍在等待验证完成 (已等待 {check_count} 秒)")
            time.sleep(1)
        
        print("✓ Cloudflare 验证已通过，继续爬取...")
        time.sleep(2)  # 额外等待页面完全加载
        
    def connect_browser(self):
        """连接到已有的浏览器"""
        print(f"正在连接到 Chrome 浏览器 (端口 9222)...")
        try:
            self.page = ChromiumPage(self.options)
            print(f"✓ 成功连接！当前页面: {self.page.url}")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            print("\n请确保：")
            print("1. 已运行 start_chrome.bat 启动 Chrome")
            print("2. Chrome 使用了 --remote-debugging-port=9222 参数")
            return False
    
    def navigate_to_search(self):
        """导航到搜索页面（如果还没在那里）"""
        if "cartoonmovement.com" not in self.page.url:
            print(f"正在导航到搜索页面...")
            self.page.get(self.search_url)
            time.sleep(3)
        
        # 检查是否需要通过验证
        self.wait_for_cloudflare("导航到搜索页面")
    
    def get_cartoon_items(self) -> list:
        """从页面中提取漫画条目（标题+图片URL）"""
        print("正在提取漫画条目...")
        
        # 先检查是否有 Cloudflare 验证
        self.wait_for_cloudflare("提取条目前")
        
        items = []
        
        try:
            articles = self.page.eles('css:article.cartoon--teaser')
            print(f"找到 {len(articles)} 个漫画条目")
            
            # 如果没有找到条目，可能是 Cloudflare 页面或加载问题
            if len(articles) == 0:
                # 再次检查是否是 Cloudflare 验证
                if self.is_cloudflare_challenge():
                    self.wait_for_cloudflare("未找到条目，重新检测")
                    # 验证通过后重新获取条目
                    articles = self.page.eles('css:article.cartoon--teaser')
                    print(f"重新检测后找到 {len(articles)} 个漫画条目")
            
            for article in articles:
                try:
                    # 提取标题 - 修正选择器，两个class在同一元素上
                    title = None
                    title_elem = article.ele('css:.field--name-title', timeout=1)
                    if title_elem:
                        title = title_elem.text.strip()
                    
                    # 提取图片URL
                    img_elem = article.ele('css:.field--name-field-media-image img', timeout=1)
                    if not img_elem:
                        img_elem = article.ele('css:img', timeout=1)
                    
                    img_url = None
                    if img_elem:
                        img_url = img_elem.attr('src') or img_elem.attr('data-src')
                        
                        if img_url:
                            if img_url.startswith('//'):
                                img_url = 'https:' + img_url
                            elif img_url.startswith('/'):
                                img_url = urljoin(self.base_url, img_url)
                    
                    if img_url and title:
                        items.append({
                            'title': title,
                            'url': img_url
                        })
                    elif img_url:
                        alt = img_elem.attr('alt') if img_elem else None
                        items.append({
                            'title': alt or f"untitled_{len(items)}",
                            'url': img_url
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"提取条目时出错: {e}")
        
        print(f"成功提取 {len(items)} 个有效条目")
        return items

    def scroll_to_load_more(self, scroll_times: int = 5, wait_time: float = 2):
        """滚动页面加载更多内容"""
        print(f"滚动页面加载更多内容（{scroll_times}次）...")
        
        for i in range(scroll_times):
            # 每次滚动前检查 Cloudflare
            self.wait_for_cloudflare(f"滚动第{i+1}次")
            
            # 缓慢滚动，让图片有时间加载
            self.page.scroll.to_bottom()
            time.sleep(wait_time)
            
            # 尝试点击"加载更多"按钮
            try:
                load_more = self.page.ele('css:button:contains("Load more")', timeout=1)
                if load_more:
                    load_more.click()
                    time.sleep(wait_time)
            except:
                pass
            
            print(f"  滚动 {i+1}/{scroll_times}")
        
        # 滚动完成后，回到顶部再慢慢滚到底部，触发所有图片加载
        print("  正在触发图片懒加载...")
        self.page.scroll.to_top()
        time.sleep(1)
        
        # 分段滚动，每次滚动一屏
        for _ in range(10):
            self.page.scroll.down(500)
            time.sleep(0.5)
        
        self.page.scroll.to_bottom()
        time.sleep(2)

    def download_image(self, url: str, title: str) -> bool:
        """下载单张图片"""
        # 检查是否已下载（通过URL检查）
        if self._is_already_downloaded(url):
            print(f"  ⊘ 已存在，跳过: {title[:50]}...")
            return False
        
        try:
            cookies = {c['name']: c['value'] for c in self.page.cookies()}
            
            headers = {
                'User-Agent': self.page.user_agent,
                'Referer': self.page.url,
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, cookies=cookies, timeout=30)
            response.raise_for_status()
            
            # 检查内容哈希（防止重复内容）
            content_hash = hashlib.md5(response.content).hexdigest()
            if content_hash in self.downloaded_content_hashes:
                print(f"  ⊘ 重复内容，跳过: {title[:50]}...")
                return False
            
            self.downloaded_content_hashes.add(content_hash)
            
            # 确定扩展名
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # 从URL中获取
                parsed = urlparse(url)
                path_ext = os.path.splitext(parsed.path)[1]
                ext = path_ext if path_ext else '.jpg'
            
            # 使用标题哈希作为文件名
            title_hash = self._get_title_hash(title)
            filename = f"{title_hash}{ext}"
            filepath = self.save_dir / filename
            
            # 如果文件已存在（哈希冲突），添加后缀
            if filepath.exists():
                counter = 1
                while filepath.exists():
                    filename = f"{title_hash}_{counter}{ext}"
                    filepath = self.save_dir / filename
                    counter += 1
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # 保存记录到CSV（增加url参数）
            self._save_record(filename, title, url)
            
            print(f"  ✓ 已保存: {filename} <- {title[:50]}...")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 下载失败 [{title[:30]}]: {e}")
            return False    
    
    def scrape_page(self) -> int:
        """爬取当前页面的图片"""
        # 爬取前检查 Cloudflare
        self.wait_for_cloudflare("爬取页面前")
        
        items = self.get_cartoon_items()
        
        downloaded = 0
        for item in items:
            if self.download_image(item['url'], item['title']):
                downloaded += 1
            time.sleep(0.3)
        
        return downloaded
    
    def go_to_page(self, page_num: int):
        """导航到指定页码"""
        current_url = self.page.url
        
        # 解析并修改URL中的页码参数
        if 'page=' in current_url:
            new_url = re.sub(r'page=\d+', f'page={page_num}', current_url)
        else:
            separator = '&' if '?' in current_url else '?'
            new_url = f"{current_url}{separator}page={page_num}"
        
        print(f"导航到第 {page_num} 页...")
        self.page.get(new_url)
        time.sleep(3)
        
        # 换页后立即检查 Cloudflare
        self.wait_for_cloudflare(f"换页到第{page_num}页后")
    
    def scrape_pages(self, start_page: int = 1, end_page: int = 10, scroll_times: int = 5):
        """爬取指定范围的页面"""
        print(f"\n开始爬取第 {start_page} 页到第 {end_page} 页...")
        print(f"已有 {len(self.downloaded_urls)} 个已下载记录")
        
        total_downloaded = 0
        
        for page_num in range(start_page, end_page + 1):
            print(f"\n{'='*50}")
            print(f"正在处理第 {page_num} 页 ({page_num - start_page + 1}/{end_page - start_page + 1})")
            print('='*50)
            
            # 导航到指定页
            self.go_to_page(page_num)
            
            # 滚动加载更多
            # self.scroll_to_load_more(scroll_times)
            
            # 爬取当前页
            downloaded = self.scrape_page()
            total_downloaded += downloaded
            
            print(f"本页下载: {downloaded} 张")
            
            # 如果本页下载数为0且不是第一次，可能有问题，额外检查一下
            if downloaded == 0 and page_num > start_page:
                print("  ⚠ 本页未下载任何图片，进行额外检查...")
                self.wait_for_cloudflare("本页无下载，额外检查")
        
        print(f"\n{'='*50}")
        print(f"爬取完成！共下载 {total_downloaded} 张图片")
        print(f"保存目录: {self.save_dir.absolute()}")
        print(f"索引文件: {self.csv_path.absolute()}")
        print('='*50)
    
    def run(self, start_page: int = 0, end_page: int = 1, scroll_times: int = 5):
        """运行爬虫"""
        if not self.connect_browser():
            return
        
        self.navigate_to_search()
        self.scrape_pages(start_page, end_page, scroll_times)
        
        print("\n爬取结束！浏览器保持打开状态。")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cartoon Movement 图片爬虫 - 接管Chrome版本')
    parser.add_argument('--save-dir', type=str, default='cartoon_images', help='图片保存目录')
    parser.add_argument('--start-page', type=int, default=1, help='起始页码') # 到第180了，第0页还没爬
    parser.add_argument('--end-page', type=int, default=1300, help='结束页码') # 到1300为止
    parser.add_argument('--scroll-times', type=int, default=1, help='每页滚动次数')
    parser.add_argument('--port', type=int, default=9222, help='Chrome调试端口')
    
    args = parser.parse_args()
    
    scraper = CartoonScraperTakeover(
        save_dir=args.save_dir,
        debug_port=args.port,
    )
    
    scraper.run(
        start_page=args.start_page,
        end_page=args.end_page,
        scroll_times=args.scroll_times
    )


if __name__ == '__main__':
    main()