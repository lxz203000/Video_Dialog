from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import numpy as np
import pickle
import json
from sklearn.metrics import mean_squared_error

class DSTC_Dialog_DataLoader(Dataset):
    """DSTC dataset loader."""
    def __init__(
            self,
            args,
            tokenizer,
            mode = "Train"
    ):
        self.features_path = args.features_path
        self.feature_framerate = args.feature_framerate
        self.max_words = args.max_words
        self.max_qa_words = args.max_qa_words
        self.max_frames = args.max_frames
        self.max_frames_vgg = args.max_frames_vgg
        self.tokenizer = tokenizer
        self.task_type = args.task_type

        if mode == "Train":
            self.dialog = self.gen_dialog(args.train_path, args.duration_path_train, args.grounding_results, args.frame_length_file, args.n_history, choice = args.choice)
            self.dialog.update(self.gen_dialog(args.val_path, args.duration_path_train, args.grounding_results, args.frame_length_file, args.n_history, choice = args.choice))
        elif mode == "Test":
            self.dialog = self.gen_dialog(args.test_path, args.duration_path_test, args.grounding_results, args.frame_length_file, args.n_history, choice = args.choice, mode=mode)
        
        self.order = []
        for i in self.dialog:
            for j in range(len(self.dialog[i])):
                self.order.append((i,j))

    def __len__(self):
        return len(self.order)

    def _get_text(self, video_id, sub_id):

        data_dict = self.dialog[video_id] 
        ind = sub_id
        total_length_with_CLS = self.max_words - 1
        total_qa_length_with_CLS = self.max_qa_words - 1
        timemask = data_dict[ind]['timemask']

        # history
        words = self.tokenizer.tokenize(data_dict[ind]['history'])
        words = ["[CLS]"] + words
        if len(words) > total_length_with_CLS:
            words = words[-total_length_with_CLS:]
        words = words + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        # question
        question_words = self.tokenizer.tokenize(data_dict[ind]['question'])
        question_words= ["[CLS]"] + question_words
        if len(question_words) > total_qa_length_with_CLS:
            question_words = question_words[-total_qa_length_with_CLS:]
        question_words = question_words + ["[SEP]"]
        input_question_ids = self.tokenizer.convert_tokens_to_ids(question_words)
        input_question_mask = [1] * len(input_question_ids)
        segment_question_ids = [0] * len(input_question_ids)
        while len(input_question_ids) < self.max_qa_words:
            input_question_ids.append(0)
            input_question_mask.append(0)
            segment_question_ids.append(0)
        assert len(input_question_ids) == self.max_qa_words
        assert len(input_question_mask) == self.max_qa_words
        assert len(segment_question_ids) == self.max_qa_words
        pairs_question_text = np.array(input_question_ids)
        pairs_question_mask = np.array(input_question_mask)
        pairs_question_segment = np.array(segment_question_ids)

        # answer
        answer_words = self.tokenizer.tokenize(data_dict[ind]['answer'])
        if len(answer_words) > total_qa_length_with_CLS:
            answer_words = answer_words[:total_qa_length_with_CLS]
        input_answer_words = ["[CLS]"] + answer_words
        output_answer_words = answer_words + ["[SEP]"]
        input_answer_ids = self.tokenizer.convert_tokens_to_ids(input_answer_words)
        output_answer_ids = self.tokenizer.convert_tokens_to_ids(output_answer_words)
        decoder_mask = [1] * len(input_answer_ids)
        while len(input_answer_ids) < self.max_qa_words:
            input_answer_ids.append(0)
            output_answer_ids.append(0)
            decoder_mask.append(0)
        assert len(input_answer_ids) == self.max_qa_words
        assert len(output_answer_ids) == self.max_qa_words
        assert len(decoder_mask) == self.max_qa_words
        pairs_input_answer_ids = np.array(input_answer_ids)
        pairs_output_answer_ids = np.array(output_answer_ids)
        pairs_decoder_mask = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, \
               pairs_input_answer_ids, pairs_decoder_mask, pairs_output_answer_ids, \
               pairs_question_text, pairs_question_mask, pairs_question_segment, \
               timemask

    def _get_video(self, vid, time_mask):

        video_mask = np.zeros((self.max_frames), dtype=np.long)
        video_mask_all = np.zeros((self.max_frames), dtype=np.long)
        vgg_mask = np.zeros((self.max_frames_vgg), dtype=np.long)
        max_video_length = 0

        s3d, vgg = load_feature(vid, self.features_path)
        video_s3d = np.zeros((self.max_frames, s3d.shape[1]), dtype=np.float)
        video_vgg = np.zeros((self.max_frames_vgg, vgg.shape[1]), dtype=np.float)

        if self.max_frames < s3d.shape[0] or self.max_frames < time_mask.shape[0]:
            s3d = s3d[:self.max_frames]
            time_mask = time_mask[:self.max_frames]

        if self.max_frames_vgg < vgg.shape[0]:
            vgg = vgg[:self.max_frames_vgg]

        slice_shape = s3d.shape
        max_video_length = max_video_length if max_video_length > slice_shape[0] else slice_shape[0]

        video_s3d[:slice_shape[0]] = s3d
        video_vgg[:vgg.shape[0]] = vgg
        vgg_mask[:vgg.shape[0]] = [1]*vgg.shape[0]

        if self.task_type == "dialog":

            video_mask[:time_mask.shape[0]] = time_mask

            idxs = range(time_mask.shape[0])
            m = np.mean(idxs)
            s = np.std(idxs)
            low = int(max(int((m-s+0)/2),0))
            high = int(min(int((m+s+time_mask.shape[0])/2),time_mask.shape[0]))
            video_mask_all[low:high] = 1

            return video_s3d, video_vgg, video_mask, vgg_mask, video_mask_all
        
        elif self.task_type == "grounding":

            video_mask[:max_video_length] = [1]*max_video_length
            time_label = np.zeros((self.max_frames),dtype=np.float)
            time_label[:time_mask.shape[0]] = time_mask
            useful_length = time_mask.shape[0]

            return video_s3d, video_vgg, video_mask, vgg_mask, time_label, useful_length
    
        else:
            raise NotImplementedError

    def __getitem__(self, feature_idx):

        video_id, sub_id = self.order[feature_idx]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_input_caption_ids, \
        pairs_decoder_mask, pairs_output_caption_ids, \
        pairs_question_text, pairs_question_mask, pairs_question_segment, \
        time_mask = self._get_text(video_id, sub_id)

        if self.task_type == "dialog":
            video_s3d, video_vgg, video_mask, vgg_mask, video_mask_all = self._get_video(video_id, time_mask)

            return pairs_text, pairs_mask, pairs_segment, video_s3d, video_vgg, video_mask, vgg_mask, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
                pairs_question_text, pairs_question_mask, pairs_question_segment, \
                video_mask_all
        
        elif self.task_type == "grounding":
            video_s3d, video_vgg, video_mask, vgg_mask, time_label, useful_length = self._get_video(video_id, time_mask)

            return pairs_text, pairs_mask, pairs_segment, video_s3d, video_vgg, video_mask, vgg_mask, \
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
                pairs_question_text, pairs_question_mask, pairs_question_segment, \
                time_label, useful_length
        
        else:
            raise NotImplementedError

    def gen_dialog(self, data_file, duration_file, grounding_results, frame_length_file, n_history, choice = "Best", mode="Train"):

        dialog_dict = dict()
        duration=dict()
        dialog_data = json.load(open(data_file, 'r'))
        all_frame_length = json.load(open(frame_length_file,'r'))

        with open(duration_file) as f:
            for line in f:
                res = line.split()
                duration[res[0][:5]] = res[1]

        if mode == "Test":
            df = open(grounding_results,'rb')
            mask_list = pickle.load(df)
            df.close() 
            mask_idx = 0 
        
        for dialog in dialog_data['dialogs']:
            vid = dialog["image_id"]
            dialog_dict[vid] = []
            frame_length = all_frame_length[vid]

            questions = [d['question'] for d in dialog['dialog']]
            answers = [d['answer'] for d in dialog['dialog']]
            real_length = float(duration[vid])
            timemasks = []

            if mode == "Train":
                for d in dialog['dialog']:
                    if d['reason']:
                        mintime = real_length
                        maxtime = 0
                        for i in d['reason']:
                            mintime = min(min(i['timestamp']), mintime)
                            maxtime = max(max(i['timestamp']), maxtime)
                        timemasks.append([mintime, maxtime])
                    else:
                        timemasks.append([0,real_length])
            else:
                for i in range(len(questions)):
                    timemasks.append(mask_list[mask_idx].detach().cpu().numpy())
                    mask_idx += 1

            for n in range(len(questions)):
                question = questions[n]
                answer = answers[n]
                cur_mask = timemasks[n]
                iou = []
                qastr = ""

                if mode =="Test" and self.task_type == "dialog":
                    if answer != "__UNDISCLOSED__":
                        continue

                if choice == "Best":
                    for i in range(n):
                        if mode == "Train":
                            iou.append(cal_iou(timemasks[n], timemasks[i]))
                        else:
                            iou.append(1-mean_squared_error(timemasks[n], timemasks[i]))
                    iou=np.asarray(iou)
                    top_idx = iou.argsort()
                    top_idx = top_idx[-1:-n_history-1:-1] 
                    top_idx.sort()
                else:
                    top_idx = range(0,n)
                    top_idx = top_idx[-n_history:]

                for i in top_idx:
                    qastr+=questions[i]
                    qastr+=" "
                    qastr+=answers[i]
                    qastr+=" "

                if self.task_type == "dialog":
                    if mode == "Train":
                        time_mask = cal_time(cur_mask, real_length, frame_length)
                    else:
                        time_mask = cal_final_mask(cur_mask)

                elif self.task_type == "grounding":
                    time_mask = cal_time(cur_mask, real_length, frame_length)

                else:
                    raise NotImplementedError

                item = {'history': qastr, 'question': question, 'answer': answer, 'timemask': time_mask}
                dialog_dict[vid].append(item)

        return dialog_dict

def load_feature(vid, path):

    vgg = np.load(path + "vggish/" + vid + ".npy")
    vgg = vgg[::2]

    s3d = pickle.load(open(path + "S3D_features/" + vid + ".pickle", "rb")).numpy()

    return s3d, vgg

def cal_iou(now, pre):
    if not now or not pre:
        return 0
    w = max(0, min(now[1], pre[1]) - max(now[0], pre[0]))
    if w == 0:
        if now[0]==pre[0]:
            return 1
        else:
            return 0
    iou = w/((now[1]-now[0])+(pre[1]-pre[0])-w)
    return iou

def cal_time(cur_mask, real_length, frame_length):
    res = np.zeros((frame_length))
    start = int(float(cur_mask[0])/real_length*frame_length)
    end = int(float(cur_mask[1])/real_length*frame_length)
    res[start:end] = 1
    return res

def cal_final_mask(cur_mask, threshold = 0.5):
    idxs = []
    for i in range(len(cur_mask)):
        if cur_mask[i] > threshold:
            idxs.append(i)
    if len(idxs) > 0:
        low = idxs[0]
        high = idxs[-1]
    else:
        low = 0
        high = len(cur_mask)-1
    for i in range(len(cur_mask)):
        if i >= low and i <= high:
            cur_mask[i] = 1
        else:
            cur_mask[i] = 0
    return cur_mask