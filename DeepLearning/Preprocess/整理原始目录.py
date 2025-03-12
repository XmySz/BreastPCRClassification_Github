import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


class DatasetOrganizer:
    def __init__(self, source_dir, target_dir, time_points=None, max_workers=4):
        """
        初始化数据集整理器

        :param source_dir: 源数据目录
        :param target_dir: 目标保存目录
        :param time_points: 时间点列表，默认为 ['T0', 'T1', 'T2', 'T3']
        :param max_workers: 并发工作线程数
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.time_points = time_points or ['T0', 'T1', 'T2', 'T3']
        self.max_workers = max_workers

        # 初始化目标目录结构
        self._initialize_target_structure()

    def _initialize_target_structure(self):
        """初始化目标目录结构"""
        for time_point in self.time_points:
            # 创建Images目录
            image_dir = self.target_dir / time_point / 'Images'
            image_dir.mkdir(parents=True, exist_ok=True)

            # 创建Labels目录
            label_dir = self.target_dir / time_point / 'Labels'
            label_dir.mkdir(parents=True, exist_ok=True)

    def _process_subject_directory(self, subject_dir):
        """
        处理单个受试者目录

        :param subject_dir: 受试者目录路径
        :return: 成功处理的文件数量
        """
        try:
            subject_name = subject_dir.name
            processed_files = 0

            # 遍历时间点目录
            for time_point in self.time_points:
                time_point_dir = subject_dir / time_point
                if not time_point_dir.exists():
                    continue

                # 处理image.nii.gz
                image_source = time_point_dir / 'image.nii.gz'
                if image_source.exists():
                    image_target = self.target_dir / time_point / 'Images' / f'{subject_name}.nii.gz'
                    shutil.copy2(image_source, image_target)
                    processed_files += 1

                # 处理label.nii.gz
                label_source = time_point_dir / 'label.nii.gz'
                if label_source.exists():
                    label_target = self.target_dir / time_point / 'Labels' / f'{subject_name}.nii.gz'
                    shutil.copy2(label_source, label_target)
                    processed_files += 1

            return processed_files

        except Exception as e:
            print(f"处理目录 {subject_dir} 时发生错误: {str(e)}")
            return 0

    def organize(self):
        """
        开始整理数据集
        """
        print(f"开始整理数据集...")
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")

        # 获取所有受试者目录
        subject_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
        total_subjects = len(subject_dirs)

        if total_subjects == 0:
            print("没有找到任何受试者目录！")
            return

        print(f"找到 {total_subjects} 个受试者目录")
        total_processed_files = 0

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务并使用tqdm显示进度
            future_to_dir = {executor.submit(self._process_subject_directory, dir_path): dir_path
                             for dir_path in subject_dirs}

            for future in tqdm(future_to_dir, total=total_subjects,
                               desc="处理进度", unit="例"):
                try:
                    processed_files = future.result()
                    total_processed_files += processed_files
                except Exception as e:
                    print(f"处理时发生错误: {str(e)}")

        print(f"\n整理完成！")
        print(f"- 共处理 {total_subjects} 个受试者目录")
        print(f"- 共复制 {total_processed_files} 个文件")


def main():
    # 设置源目录和目标目录
    source_dir = r"F:\Data\HX\Dataset\ISPY2_SELECT\Original"
    target_dir = r"F:\Data\HX\Dataset\ISPY2_SELECT\Organized"

    # 创建整理器并执行整理
    organizer = DatasetOrganizer(
        source_dir=source_dir,
        target_dir=target_dir,
        time_points=['T0', 'T1', 'T2', 'T3'],
        max_workers=4
    )

    organizer.organize()


if __name__ == "__main__":
    main()
